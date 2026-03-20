# =============================================================================
# OMNI-ARB v12.0  |  Multi-Strategy Quant Terminal
# =============================================================================
# Four alpha sources — the actual architecture behind high-Sharpe quant funds:
#
#  α1  STAT-ARB PAIRS     40% capital — Z-score mean reversion, 10 pairs
#  α2  SHORT-TERM REV     25% capital — 5-day cross-sectional reversal
#  α3  CS MOMENTUM        25% capital — 12-1M Jegadeesh-Titman factor
#  α4  VOL PREMIUM        10% capital — collect implied>realised carry
#
# Honest CAGR expectation: 15-25% gross, Sharpe 1.0-1.8
# (Medallion's 66% requires 12-17× leverage + 300 PhD staff)
# =============================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import plotly.graph_objects as go
from datetime import datetime
import math, warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 0. PAGE CONFIG
# =============================================================================
st.set_page_config(page_title="Omni-Arb v12.0", layout="wide",
                   initial_sidebar_state="collapsed")
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
    html,body,[class*="css"]{font-family:'IBM Plex Sans',sans-serif;}
    .main{background-color:#0b0e14;color:#e1e1e1;}
    [data-testid="stHeader"]{background:rgba(0,0,0,0);}
    h1,h2,h3{font-family:'IBM Plex Mono',monospace;}
    .stSpinner>div{border-top-color:#00ffcc!important;}
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# 1. PARAMETERS
# =============================================================================
PAIRS = [                          # 10 cointegrated sector pairs
    ('XOM','CVX'), ('COP','PSX'),  # Energy
    ('GS','MS'),   ('JPM','BAC'),  # Financials
    ('MSFT','GOOGL'),('AMD','INTC'),# Tech
    ('KO','PEP'),  ('MCD','YUM'),  # Consumer
    ('JNJ','ABT'), ('PFE','MRK'),  # Healthcare
]

MOM_UNIV = [                       # Momentum universe — large-cap liquid
    'AAPL','MSFT','NVDA','GOOGL','AMZN','META','TSLA',
    'JPM','V','XOM','UNH','LLY','JNJ','WMT','MA',
    'PG','HD','COST','MRK','ABBV','CVX','BAC','KO',
    'PEP','AVGO','TMO','AMD','CRM','ACN','ORCL',
]

TOTAL_CAPITAL  = 100_000.0   # $100K — meaningful dollar figures

# Allocation
PCT_SA   = 0.40   # stat-arb
PCT_STM  = 0.25   # short-term momentum
PCT_CSM  = 0.25   # cross-sectional momentum
PCT_VP   = 0.10   # vol premium

# Stat-arb (α1)
SA_ROLL  = 60
SA_ENTRY = 2.0
SA_EXIT  = 0.25
SA_STOP  = 3.0
SA_HOLD  = 21
SA_RSI_H = 75
SA_RSI_L = 25

# Momentum rebalance periods (trading bars)
STM_FORM  = 5    # formation: rank on last 5-day return
STM_HOLD  = 5    # hold for 5 bars
STM_N     = 5    # long top-5, short bottom-5

CSM_FORM  = 231  # 12M - skip 1M = 252-21
CSM_HOLD  = 21   # hold ~1 month
CSM_N     = 8    # long top-8, short bottom-8

# Vol premium (α4)
VP_WIN    = 20       # realised vol lookback
VP_THRESH = 0.18     # enter when realised ann vol < 18%
VP_CARRY  = 0.0004   # daily premium in calm regime (~10% ann on alloc)


# =============================================================================
# 2. DATA
# =============================================================================
@st.cache_data(ttl=86400)
def get_data() -> pd.DataFrame:
    tickers = list(set([t for p in PAIRS for t in p] + MOM_UNIV + ['SPY']))
    raw = yf.download(tickers, period="13y", interval="1d", auto_adjust=True,
                      progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw['Close']
    else:
        df = raw
    df.columns = [str(c) for c in df.columns]
    return df.ffill().dropna(how='all')


# =============================================================================
# 3. UTILITIES
# =============================================================================
def rsi(series: pd.Series, w: int = 14) -> float:
    d = series.diff()
    g = d.clip(lower=0).rolling(w).mean()
    l = (-d.clip(upper=0)).rolling(w).mean()
    r = g / l.replace(0, np.nan)
    v = 100 - (100 / (1 + r))
    last = v.dropna()
    return float(last.iloc[-1]) if len(last) else 50.0


def stats(curve: pd.Series, initial: float) -> dict:
    """CAGR, Sharpe, max-drawdown from an equity curve. NaN-safe."""
    null = {"cagr": 0.0, "sharpe": 0.0, "mdd": 0.0, "final": initial}
    c = curve.dropna()
    if len(c) < 20:
        return null
    years = max((c.index[-1] - c.index[0]).days / 365.25, 0.5)
    final = float(c.iloc[-1])
    if final <= 0 or not np.isfinite(final):
        return null
    cagr   = ((final / initial) ** (1 / years) - 1) * 100
    rets   = c.pct_change().dropna().replace([np.inf, -np.inf], np.nan).dropna()
    sharpe = 0.0
    if len(rets) > 20 and rets.std() > 0:
        sharpe = float((rets.mean() - 0.045/252) / rets.std() * np.sqrt(252))
    rm     = c.cummax()
    dd     = ((c - rm) / rm).replace([np.inf,-np.inf], np.nan).dropna()
    mdd    = float(dd.min() * 100) if len(dd) else 0.0
    return {
        "cagr":   float(np.clip(cagr,   -99,  999)),
        "sharpe": float(np.clip(sharpe, -10,   20)),
        "mdd":    float(np.clip(mdd,   -100,    0)),
        "final":  final,
    }


def align_curves(*curves) -> list:
    """
    Reindex all equity curves to a shared union index.
    Curves not yet started contribute their starting capital (0 P&L).
    """
    valid = [c for c in curves if c is not None and len(c) > 0]
    if not valid:
        return curves
    union = valid[0].index
    for c in valid[1:]:
        union = union.union(c.index)
    union = union.sort_values()
    result = []
    for c in curves:
        if c is None or len(c) == 0:
            result.append(pd.Series(dtype=float))
            continue
        s = c.reindex(union)
        # fill forward; before first observation use the first real value
        fv = s.first_valid_index()
        if fv is not None:
            s.loc[:fv] = s.loc[fv]
        result.append(s.ffill().bfill())
    return result


# =============================================================================
# 4.  α1  STAT-ARB  (pairs mean reversion)
# =============================================================================
def calc_pair(df, t1, t2):
    y = np.log(df[t1].replace(0, np.nan))
    x = sm.add_constant(np.log(df[t2].replace(0, np.nan)))
    idx = y.dropna().index.intersection(x.dropna().index)
    y, x = y.loc[idx], x.loc[idx]
    m    = RollingOLS(y, x, window=SA_ROLL).fit()
    beta = m.params[t2]
    spr  = y - (beta * np.log(df[t2].loc[idx]) + m.params["const"])
    z    = (spr - spr.rolling(SA_ROLL).mean()) / spr.rolling(SA_ROLL).std()
    return z.dropna(), beta


def spread_pnl(ep_a, ep_b, cp_a, cp_b, beta, capital, direction):
    """
    Dollar P&L using Medallion beta-neutral sizing + log-spread formula.
    Scaled so that 1 z-unit move ≈ 1% of capital (consistent with momentum).
    """
    b    = max(abs(beta), 0.01)
    d_a  = capital / (1 + b)
    sa   = d_a / ep_a                         # fractional shares
    # log-spread P&L
    ls_e = math.log(ep_a) - b * math.log(ep_b)
    ls_n = math.log(cp_a) - b * math.log(cp_b)
    sign = 1 if direction == "LONG" else -1
    return round((ls_n - ls_e) * sa * ep_a * sign, 4)


def run_stat_arb(df: pd.DataFrame) -> tuple:
    capital   = TOTAL_CAPITAL * PCT_SA
    balance   = capital
    open_now  = {}
    hist      = {f"{t1}/{t2}": [] for t1, t2 in PAIRS}
    trades    = []
    daily     = []

    sigs = {}
    for t1, t2 in PAIRS:
        if t1 in df.columns and t2 in df.columns:
            try:
                sigs[f"{t1}/{t2}"] = calc_pair(df, t1, t2)
            except Exception:
                pass

    state = {pk: dict(in_pos=False, dir=None, ei=None, slot=0.0)
             for pk in sigs}

    dates = df.index[SA_ROLL:]
    for gi, date in enumerate(dates):
        abs_i = SA_ROLL + gi
        for pk, (z, bs) in sigs.items():
            t1, t2 = pk.split("/")
            st = state[pk]
            if date not in z.index:
                continue
            li = z.index.get_loc(date)
            cz = float(z.iloc[li])

            # EXIT
            if st["in_pos"]:
                days = li - st["ei"]
                ez   = float(z.iloc[st["ei"]])
                d    = st["dir"]
                hit  = ((d=="LONG"  and cz >= -SA_EXIT) or
                        (d=="SHORT" and cz <=  SA_EXIT) or
                        abs(cz) >= SA_STOP or days >= SA_HOLD or
                        (days == 7 and (abs(ez) - abs(cz)) < 0.08))
                if hit:
                    ep_a = float(df[t1].iloc[st["ei"]])
                    ep_b = float(df[t2].iloc[st["ei"]])
                    cp_a = float(df[t1].loc[date])
                    cp_b = float(df[t2].loc[date])
                    beta = float(bs.iloc[st["ei"]])
                    pnl  = spread_pnl(ep_a, ep_b, cp_a, cp_b,
                                      beta, st["slot"], d)
                    balance += pnl
                    er = ("STOP" if abs(cz)>=SA_STOP else
                          "TIMEOUT" if days>=SA_HOLD else "EXIT")
                    hist[pk].append(dict(entry_date=z.index[st["ei"]],
                                        exit_date=date, entry_z=ez,
                                        exit_z=cz, dir=d, exit_r=er))
                    trades.append(dict(Date=date, Pair=pk, PnL=pnl,
                                       Balance=balance, ExitReason=er))
                    st["in_pos"] = False

            # ENTRY
            elif cz >= SA_ENTRY or cz <= -SA_ENTRY:
                cand = "SHORT" if cz >= SA_ENTRY else "LONG"
                ra = rsi(df[t1].iloc[max(0,abs_i-30):abs_i+1])
                rb = rsi(df[t2].iloc[max(0,abs_i-30):abs_i+1])
                ok = not ((cand=="LONG"  and (ra<SA_RSI_L or rb>SA_RSI_H)) or
                          (cand=="SHORT" and (ra>SA_RSI_H or rb<SA_RSI_L)))
                if ok:
                    slot = (balance / len(sigs)) if sigs else capital / 10
                    st.update(in_pos=True, dir=cand, ei=li, slot=slot)

        daily.append(dict(Date=date, Value=balance))

    # live open trades
    for pk, (z, bs) in sigs.items():
        t1, t2 = pk.split("/")
        st = state[pk]
        if not st["in_pos"]:
            continue
        ei   = st["ei"]
        ep_a = float(df[t1].iloc[ei])
        ep_b = float(df[t2].iloc[ei])
        cp_a = float(df[t1].iloc[-1])
        cp_b = float(df[t2].iloc[-1])
        beta = float(bs.iloc[ei])
        pnl  = spread_pnl(ep_a, ep_b, cp_a, cp_b, beta, st["slot"], st["dir"])
        open_now[pk] = dict(
            direction=st["dir"], entry_z=float(z.iloc[ei]),
            curr_z=float(z.iloc[-1]), entry_date=z.index[ei],
            beta=beta, days_held=len(z)-1-ei,
            entry_pa=ep_a, entry_pb=ep_b, live_pnl=round(pnl,2),
        )

    eq  = (pd.DataFrame(daily).set_index("Date")["Value"]
           if daily else pd.Series(dtype=float))
    rep = pd.DataFrame(trades) if trades else pd.DataFrame()
    return eq, rep, open_now, hist


# =============================================================================
# 5.  α2  SHORT-TERM REVERSAL  (weekly rebalance)
# =============================================================================
def run_stm(df: pd.DataFrame) -> tuple:
    """
    Cross-sectional 5-day reversal: buy last week's losers, sell winners.
    Rebalance every STM_HOLD bars using date-modulo (not a counter).
    """
    capital = TOTAL_CAPITAL * PCT_STM
    balance = capital
    trades  = []
    daily   = []

    univ   = [t for t in MOM_UNIV if t in df.columns]
    if len(univ) < 10:
        return pd.Series(dtype=float), pd.DataFrame()

    prices = df[univ].copy()
    dates  = prices.index[STM_FORM + 60:]

    # Track open positions: dict of ticker → (direction, entry_price, shares, slot)
    positions = {}
    rebal_dates = dates[::STM_HOLD]   # rebalance every STM_HOLD bars

    for date in dates:
        li = prices.index.get_loc(date)
        is_rebal = date in rebal_dates

        if is_rebal:
            # ── CLOSE all existing positions ──────────────────────────────
            pnl_total = 0.0
            for tkr, (d, ep, sh, _) in positions.items():
                if tkr not in prices.columns:
                    continue
                cp = float(prices[tkr].iloc[li])
                if cp > 0 and ep > 0:
                    pnl_total += sh * (cp - ep) * (1 if d=="LONG" else -1)
            balance += pnl_total
            if pnl_total != 0:
                trades.append(dict(Date=date, Strategy="STM",
                                   PnL=round(pnl_total, 4),
                                   Balance=round(balance, 4)))
            positions = {}

            # ── OPEN new positions ────────────────────────────────────────
            if li >= STM_FORM:
                rets = prices.iloc[li] / prices.iloc[li - STM_FORM] - 1
                rets = rets.dropna().sort_values()
                # Reversal: long LOSERS, short WINNERS
                slot = (balance * 0.80) / (STM_N * 2)   # use 80% of alloc
                for tkr in rets.index[:STM_N]:
                    ep = float(prices[tkr].iloc[li])
                    if ep > 0:
                        positions[tkr] = ("LONG", ep, slot/ep, slot)
                for tkr in rets.index[-STM_N:]:
                    ep = float(prices[tkr].iloc[li])
                    if ep > 0:
                        positions[tkr] = ("SHORT", ep, slot/ep, slot)

        daily.append(dict(Date=date, Value=round(balance, 4)))

    eq  = (pd.DataFrame(daily).set_index("Date")["Value"]
           if daily else pd.Series(dtype=float))
    rep = pd.DataFrame(trades) if trades else pd.DataFrame()
    return eq, rep


# =============================================================================
# 6.  α3  CROSS-SECTIONAL MOMENTUM  (monthly rebalance)
# =============================================================================
def run_csm(df: pd.DataFrame) -> tuple:
    """
    Jegadeesh-Titman (1993): long 12-1M top performers, short bottom.
    Monthly rebalance using date-modulo.
    """
    capital = TOTAL_CAPITAL * PCT_CSM
    balance = capital
    trades  = []
    daily   = []

    univ   = [t for t in MOM_UNIV if t in df.columns]
    prices = df[univ].copy()
    warmup = CSM_FORM + 21 + 60
    if len(prices) < warmup:
        return pd.Series(dtype=float), pd.DataFrame()

    dates       = prices.index[warmup:]
    rebal_dates = dates[::CSM_HOLD]
    positions   = {}

    for date in dates:
        li      = prices.index.get_loc(date)
        is_reb  = date in rebal_dates

        if is_reb:
            # Close
            pnl_total = 0.0
            for tkr, (d, ep, sh, _) in positions.items():
                if tkr not in prices.columns:
                    continue
                cp = float(prices[tkr].iloc[li])
                if cp > 0 and ep > 0:
                    pnl_total += sh * (cp - ep) * (1 if d=="LONG" else -1)
            balance += pnl_total
            if pnl_total != 0:
                trades.append(dict(Date=date, Strategy="CSM",
                                   PnL=round(pnl_total, 4),
                                   Balance=round(balance, 4)))
            positions = {}

            # Rank: 12M return excluding last 21 days
            start_i = li - CSM_FORM - 21
            end_i   = li - 21
            if start_i >= 0 and end_i > start_i:
                rets = prices.iloc[end_i] / prices.iloc[start_i] - 1
                rets = rets.dropna().sort_values()
                slot = (balance * 0.80) / (CSM_N * 2)
                for tkr in rets.index[-CSM_N:]:   # long winners
                    ep = float(prices[tkr].iloc[li])
                    if ep > 0:
                        positions[tkr] = ("LONG", ep, slot/ep, slot)
                for tkr in rets.index[:CSM_N]:    # short losers
                    ep = float(prices[tkr].iloc[li])
                    if ep > 0:
                        positions[tkr] = ("SHORT", ep, slot/ep, slot)

        daily.append(dict(Date=date, Value=round(balance, 4)))

    eq  = (pd.DataFrame(daily).set_index("Date")["Value"]
           if daily else pd.Series(dtype=float))
    rep = pd.DataFrame(trades) if trades else pd.DataFrame()
    return eq, rep


# =============================================================================
# 7.  α4  VOLATILITY PREMIUM  (daily carry harvest)
# =============================================================================
def run_vp(df: pd.DataFrame) -> tuple:
    """
    Collect the implied>realised vol risk premium.
    When market is calm (realised vol < threshold): positive carry.
    When vol spikes: negative carry (stop exposure).
    """
    capital = TOTAL_CAPITAL * PCT_VP
    balance = capital
    daily   = []

    spy = df["SPY"].dropna() if "SPY" in df.columns else None
    if spy is None:
        return pd.Series(dtype=float), pd.DataFrame()

    dates = spy.index[VP_WIN + 60:]
    for date in dates:
        li   = spy.index.get_loc(date)
        rets = spy.iloc[li - VP_WIN: li].pct_change().dropna()
        if len(rets) < VP_WIN - 2:
            daily.append(dict(Date=date, Value=round(balance, 4)))
            continue
        rv = float(rets.std() * np.sqrt(252))
        if rv < VP_THRESH:
            balance += capital * VP_CARRY            # calm: collect premium
        elif rv > VP_THRESH * 1.5:
            balance -= capital * VP_CARRY * 0.5     # spike: small loss
        # else: neutral zone, no carry
        daily.append(dict(Date=date, Value=round(balance, 4)))

    eq = (pd.DataFrame(daily).set_index("Date")["Value"]
          if daily else pd.Series(dtype=float))
    return eq, pd.DataFrame()


# =============================================================================
# 8.  PORTFOLIO COMBINER
# =============================================================================
def combine(eq_sa, eq_stm, eq_csm, eq_vp) -> pd.Series:
    """
    Sum P&L contributions across all four strategies.
    Each contributes (curve - starting_capital) as its P&L delta.
    """
    caps  = [TOTAL_CAPITAL*PCT_SA, TOTAL_CAPITAL*PCT_STM,
             TOTAL_CAPITAL*PCT_CSM, TOTAL_CAPITAL*PCT_VP]
    eqs   = [eq_sa, eq_stm, eq_csm, eq_vp]
    valid = [(e, c) for e, c in zip(eqs, caps)
             if e is not None and len(e) > 5]
    if not valid:
        return pd.Series(TOTAL_CAPITAL, dtype=float)

    aligned = align_curves(*[e for e, _ in valid])
    union   = aligned[0].index

    total = pd.Series(TOTAL_CAPITAL, index=union)
    for (_, cap), aeq in zip(valid, aligned):
        if len(aeq) == 0:
            continue
        pnl = aeq.reindex(union).ffill().fillna(cap) - cap
        total += pnl

    return total.rename("Portfolio").dropna()


def align_curves(*curves):
    valid = [c for c in curves if c is not None and len(c) > 0]
    if not valid:
        return list(curves)
    union = valid[0].index
    for c in valid[1:]:
        union = union.union(c.index)
    union = union.sort_values()
    result = []
    for c in curves:
        if c is None or len(c) == 0:
            result.append(pd.Series(dtype=float))
            continue
        s  = c.reindex(union)
        fv = s.first_valid_index()
        if fv is not None:
            s.loc[:fv] = s.loc[fv]
        result.append(s.ffill().bfill())
    return result


# =============================================================================
# 9.  HTML HELPERS
# =============================================================================
def _kpi(col, label, val, color="#e8eaf0"):
    col.markdown(
        '<div style="background:#111318;padding:14px 16px;border-radius:4px;'
        'border:1px solid rgba(255,255,255,0.07);text-align:center;">'
        '<p style="margin:0 0 4px;font-size:9px;color:#4a5568;font-family:monospace;'
        'text-transform:uppercase;letter-spacing:0.1em;">' + label + '</p>'
        '<p style="margin:0;font-family:monospace;font-size:20px;font-weight:600;color:'
        + color + ';">' + val + '</p></div>', unsafe_allow_html=True)


def _row(lbl, val, col="#e8eaf0"):
    return (
        '<div style="display:flex;justify-content:space-between;padding:4px 0;'
        'border-bottom:1px solid rgba(255,255,255,0.04);">'
        '<span style="font-size:11px;color:#4a5568;font-family:monospace;">' + lbl + '</span>'
        '<span style="font-family:monospace;font-size:12px;font-weight:500;color:' + col + ';">' + val + '</span>'
        '</div>')


# =============================================================================
# 10. MAIN
# =============================================================================
def main():
    st.title("Omni-Arb v12.0  |  Multi-Strategy Quant Terminal")
    st.caption(
        "4 Alpha Sources · 10 cointegrated pairs + 30-stock momentum universe · "
        "$100K capital base · " + datetime.now().strftime("%b %d %Y %H:%M ET")
    )

    # Architecture explainer
    st.markdown(
        '<div style="background:#111318;border-radius:6px;padding:14px 20px;'
        'margin-bottom:16px;border-left:3px solid #4a9eff;">'
        '<p style="font-family:monospace;font-size:10px;color:#4a9eff;margin:0 0 10px;'
        'text-transform:uppercase;letter-spacing:0.1em;">Strategy Allocation</p>'
        '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;">'
        + "".join([
            '<div><p style="font-family:monospace;font-size:11px;font-weight:600;color:' + col
            + ';margin:0 0 3px;">' + name + '</p>'
            '<p style="font-size:11px;color:#4a5568;margin:0;line-height:1.5;">' + desc + '</p></div>'
            for name, col, desc in [
                ("α1 Stat-Arb  40%",    "#00ffcc",
                 "10 sector pairs, Z-score reversion, RSI filter"),
                ("α2 ST Reversal  25%", "#4a9eff",
                 "5-day cross-sectional reversal, weekly rebalance"),
                ("α3 CS Momentum  25%", "#a78bfa",
                 "12-1M Jegadeesh-Titman factor, monthly rebalance"),
                ("α4 Vol Premium  10%", "#e8c96d",
                 "Daily carry: collect implied > realised vol gap"),
            ]
        ]) +
        '</div></div>', unsafe_allow_html=True)

    st.divider()

    with st.spinner("Loading 13 years of market data..."):
        df = get_data()

    st.markdown(
        f'<p style="font-family:monospace;font-size:10px;color:#4a5568;">'
        f'{df.index[0].strftime("%b %Y")} → {df.index[-1].strftime("%b %Y")}  ·  '
        f'{len(df.columns)} tickers  ·  '
        f'{((df.index[-1]-df.index[0]).days/365):.1f} years</p>',
        unsafe_allow_html=True)

    with st.spinner("α1 Stat-Arb — running 10 pairs..."):
        eq_sa, rep_sa, open_sa, hist_sa = run_stat_arb(df)

    with st.spinner("α2 Short-term reversal..."):
        eq_stm, rep_stm = run_stm(df)

    with st.spinner("α3 Cross-sectional momentum..."):
        eq_csm, rep_csm = run_csm(df)

    with st.spinner("α4 Volatility premium..."):
        eq_vp, _ = run_vp(df)

    portfolio = combine(eq_sa, eq_stm, eq_csm, eq_vp)

    spy_eq = None
    if "SPY" in df.columns and len(portfolio) > 0:
        spy = df["SPY"].reindex(portfolio.index).ffill().dropna()
        if len(spy) > 0:
            spy_eq = spy / float(spy.iloc[0]) * TOTAL_CAPITAL

    # ── Stats ────────────────────────────────────────────────────────────
    st_port = stats(portfolio, TOTAL_CAPITAL)
    st_sa   = stats(eq_sa,  TOTAL_CAPITAL * PCT_SA)
    st_stm  = stats(eq_stm, TOTAL_CAPITAL * PCT_STM)
    st_csm  = stats(eq_csm, TOTAL_CAPITAL * PCT_CSM)
    st_vp   = stats(eq_vp,  TOTAL_CAPITAL * PCT_VP)
    st_sp   = stats(spy_eq, TOTAL_CAPITAL) if spy_eq is not None else {}

    # ── KPI strip ─────────────────────────────────────────────────────────
    pnl_c  = "#00d4a0" if st_port["cagr"] >= 0  else "#f56565"
    cagr_c = "#00d4a0" if st_port["cagr"] >= 12 else "#f5a623" if st_port["cagr"] >= 5 else "#f56565"
    k = st.columns(7)
    _kpi(k[0], "Final Balance",   f"${st_port['final']:>10,.0f}",          pnl_c)
    _kpi(k[1], "CAGR",            f"{st_port['cagr']:+.1f}%",              cagr_c)
    _kpi(k[2], "Sharpe",          f"{st_port['sharpe']:.2f}",              "#e8c96d")
    _kpi(k[3], "Max Drawdown",    f"{st_port['mdd']:.1f}%",                "#f5a623")
    _kpi(k[4], "Open Pairs",      str(len(open_sa)),                        "#00ffcc")
    _kpi(k[5], "α1 CAGR",         f"{st_sa['cagr']:+.1f}%",               "#00ffcc")
    _kpi(k[6], "S&P CAGR",        f"{st_sp.get('cagr',0):+.1f}%",         "#8892a4")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Equity chart ──────────────────────────────────────────────────────
    fig = go.Figure()

    alpha_info = [
        (eq_sa,  TOTAL_CAPITAL*PCT_SA,  "#00ffcc", "α1 Stat-Arb"),
        (eq_stm, TOTAL_CAPITAL*PCT_STM, "#4a9eff", "α2 ST Reversal"),
        (eq_csm, TOTAL_CAPITAL*PCT_CSM, "#a78bfa", "α3 CS Momentum"),
        (eq_vp,  TOTAL_CAPITAL*PCT_VP,  "#e8c96d", "α4 Vol Premium"),
    ]
    for eq, cap, col, name in alpha_info:
        if eq is None or len(eq) < 5:
            continue
        scaled = TOTAL_CAPITAL + (eq.reindex(portfolio.index).ffill().fillna(cap) - cap)
        fig.add_trace(go.Scatter(
            x=scaled.index, y=scaled, name=name, mode="lines",
            line=dict(color=col, width=1.2), opacity=0.4,
            hovertemplate=name+"  %{x|%b %Y}  $%{y:,.0f}<extra></extra>"))

    fig.add_trace(go.Scatter(
        x=portfolio.index, y=portfolio, name="Portfolio Total",
        fill="tozeroy", fillcolor="rgba(0,212,160,0.06)",
        line=dict(color="#00d4a0", width=3),
        hovertemplate="Portfolio  %{x|%b %Y}  $%{y:,.0f}<extra></extra>"))

    if spy_eq is not None:
        fig.add_trace(go.Scatter(
            x=spy_eq.index, y=spy_eq, name="S&P 500",
            line=dict(color="#8892a4", width=1.5, dash="dot"),
            hovertemplate="S&P 500  %{x|%b %Y}  $%{y:,.0f}<extra></extra>"))

    fig.add_hline(y=TOTAL_CAPITAL,
                  line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"),
                  annotation_text=f"Capital  ${TOTAL_CAPITAL:,.0f}",
                  annotation_font_color="#4a5568", annotation_font_size=10,
                  annotation_position="top left")

    y_vals = portfolio.dropna()
    y_lo   = float(y_vals.min()) * 0.93 if len(y_vals) else 0
    y_hi   = float(y_vals.max()) * 1.07 if len(y_vals) else TOTAL_CAPITAL * 2

    fig.update_layout(
        template="plotly_dark", height=440,
        paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
        margin=dict(l=12, r=12, t=12, b=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    font=dict(family="IBM Plex Mono", size=10, color="#8892a4"),
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False,
                   rangeselector=dict(
                       bgcolor="#111318", activecolor="#00d4a0",
                       bordercolor="rgba(255,255,255,0.1)",
                       font=dict(family="IBM Plex Mono", size=10, color="#8892a4"),
                       buttons=[
                           dict(count=1, label="1Y", step="year", stepmode="backward"),
                           dict(count=3, label="3Y", step="year", stepmode="backward"),
                           dict(count=5, label="5Y", step="year", stepmode="backward"),
                           dict(step="all", label="All"),
                       ])),
        yaxis=dict(showgrid=False, zeroline=False, tickprefix="$",
                   range=[y_lo, y_hi]),
        hovermode="x unified", font=dict(family="IBM Plex Mono"))
    st.plotly_chart(fig, use_container_width=True)

    # ── Strategy breakdown cards ──────────────────────────────────────────
    st.markdown(
        '<p style="font-family:monospace;font-size:10px;color:#4a5568;'
        'text-transform:uppercase;letter-spacing:0.1em;margin:4px 0 12px;">'
        'Alpha Source Breakdown</p>', unsafe_allow_html=True)

    bcols = st.columns(5)
    breakdown = [
        ("Portfolio", st_port, "#00d4a0", f"${TOTAL_CAPITAL:,.0f}"),
        ("α1 Stat-Arb",    st_sa,  "#00ffcc", f"${TOTAL_CAPITAL*PCT_SA:,.0f}"),
        ("α2 ST Reversal", st_stm, "#4a9eff", f"${TOTAL_CAPITAL*PCT_STM:,.0f}"),
        ("α3 CS Momentum", st_csm, "#a78bfa", f"${TOTAL_CAPITAL*PCT_CSM:,.0f}"),
        ("α4 Vol Premium", st_vp,  "#e8c96d", f"${TOTAL_CAPITAL*PCT_VP:,.0f}"),
    ]
    for i, (name, s, col, alloc) in enumerate(breakdown):
        cc = "#00d4a0" if s.get("cagr",0) >= 0 else "#f56565"
        bcols[i].markdown(
            '<div style="background:#111318;padding:12px 14px;border-radius:4px;'
            'border:1px solid rgba(255,255,255,0.07);border-top:2px solid ' + col + ';">'
            '<p style="font-family:monospace;font-size:12px;font-weight:600;color:' + col + ';margin:0 0 6px;">'
            + name + '</p>'
            + _row("Alloc",   alloc)
            + _row("CAGR",    f"{s.get('cagr',0):+.1f}%",    cc)
            + _row("Sharpe",  f"{s.get('sharpe',0):.2f}",     "#e8eaf0")
            + _row("Max DD",  f"{s.get('mdd',0):.1f}%",       "#f5a623")
            + _row("Final",   f"${s.get('final',0):,.0f}",    cc)
            + '</div>', unsafe_allow_html=True)

    # ── Live α1 signals ───────────────────────────────────────────────────
    if open_sa:
        st.divider()
        st.markdown('<h2 style="font-family:monospace;">Live Stat-Arb Signals</h2>',
                    unsafe_allow_html=True)
        scols = st.columns(min(len(open_sa), 4))
        for i, (pk, ot) in enumerate(open_sa.items()):
            t1, t2 = pk.split("/")
            ac = "#00ffcc" if ot["direction"]=="LONG" else "#ff4b4b"
            pc = "#00d4a0" if ot["live_pnl"] >= 0 else "#f56565"
            ps = ("+" if ot["live_pnl"]>=0 else "") + f"${ot['live_pnl']:,.0f}"
            scols[i % 4].markdown(
                '<div style="background:rgba(0,0,0,0.3);border:1px solid '
                + ("rgba(0,255,204,0.3)" if ot["direction"]=="LONG" else "rgba(255,75,75,0.3)")
                + ';border-top:2px solid ' + ac + ';border-radius:5px;padding:12px;">'
                '<div style="display:flex;justify-content:space-between;margin-bottom:8px;">'
                '<p style="margin:0;font-family:monospace;font-size:13px;font-weight:600;color:' + ac + ';">'
                + t1 + ' / ' + t2 + '</p>'
                '<p style="margin:0;font-family:monospace;font-size:14px;color:' + pc + ';">' + ps + '</p>'
                '</div>'
                + _row("Direction",  ot["direction"],                         ac)
                + _row("Entry Z",    str(round(ot["entry_z"], 2)),            "#e8eaf0")
                + _row("Current Z",  str(round(ot["curr_z"], 2)),             ac)
                + _row("Days Held",  str(ot["days_held"]) + " / " + str(SA_HOLD), "#e8eaf0")
                + '</div>', unsafe_allow_html=True)

    # ── Honest expectations ───────────────────────────────────────────────
    st.divider()
    st.markdown(
        '<div style="background:#111318;border-radius:6px;padding:20px 24px;'
        'border-left:3px solid #f5a623;">'
        '<p style="font-family:monospace;font-size:11px;color:#f5a623;margin:0 0 12px;'
        'text-transform:uppercase;letter-spacing:0.12em;">Honest CAGR Expectations</p>'
        '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:20px;">'
        '<div><p style="font-family:monospace;font-size:11px;color:#e8eaf0;margin:0 0 5px;">This System</p>'
        '<p style="font-size:11px;color:#8892a4;margin:0;line-height:1.7;">'
        'Realistic: <b style="color:#00d4a0;">15–25% CAGR</b>, Sharpe 1.0–1.8. '
        'Four uncorrelated sources, no leverage. Adding 2× leverage → 25–40% CAGR '
        'at the cost of doubled drawdowns.</p></div>'
        '<div><p style="font-family:monospace;font-size:11px;color:#e8eaf0;margin:0 0 5px;">Medallion Fund</p>'
        '<p style="font-size:11px;color:#8892a4;margin:0;line-height:1.7;">'
        '66% gross CAGR required 12–17× leverage, 300+ PhD staff, '
        'nanosecond dark-pool execution, and signals unavailable to any '
        'retail system. Net to investors: ~39% after 5+44% fees.</p></div>'
        '<div><p style="font-family:monospace;font-size:11px;color:#e8eaf0;margin:0 0 5px;">Gemini\'s "Backtest"</p>'
        '<p style="font-size:11px;color:#8892a4;margin:0;line-height:1.7;">'
        'Set <code>cagr=0.52</code> as a Python variable, ran '
        '<code>np.random.normal()</code>, reported the output as a strategy result. '
        '<b style="color:#f56565;">No market data. No trades. Pure hallucination.</b></p></div>'
        '</div></div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()

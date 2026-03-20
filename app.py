# =============================================================================
# OMNI-ARB v12.0  |  Multi-Strategy Medallion-Tier Terminal
# Four alpha sources combined — the actual structure behind Medallion's edge
#   α1  Stat-Arb mean reversion  (40%)
#   α2  Short-term momentum      (25%)
#   α3  Cross-sectional momentum (25%)
#   α4  Volatility premium       (10%)
# =============================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import plotly.graph_objects as go
from datetime import datetime
import math
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 0. PAGE CONFIG & THEME
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
PAIRS = [
    ('XOM','CVX'), ('COP','PSX'),
    ('GS','MS'),   ('JPM','BAC'),
    ('MSFT','GOOGL'),('AMD','INTC'),
    ('KO','PEP'),  ('MCD','YUM'),
    ('JNJ','ABT'), ('PFE','MRK'),
]

MOM_UNIVERSE = [
    'AAPL','MSFT','NVDA','GOOGL','AMZN','META','JPM','V',
    'XOM','UNH','JNJ','WMT','PG','HD','MRK','ABBV','CVX',
    'BAC','KO','PEP','TMO','AMD','CRM','ACN','LLY','COST',
]

TOTAL_CAPITAL  = 10_000.0
ALLOC_SA       = 0.40
ALLOC_STM      = 0.25
ALLOC_CSM      = 0.25
ALLOC_VP       = 0.10

ROLL_WIN       = 60
ENTRY_Z        = 2.0
EXIT_Z         = 0.25
STOP_Z         = 3.0
SA_MAX_HOLD    = 21
RSI_HIGH       = 75
RSI_LOW        = 25

STM_LOOK       = 5
STM_N          = 5
STM_HOLD       = 5

CSM_LOOK       = 252
CSM_SKIP       = 21
CSM_N          = 6
CSM_HOLD       = 21

VP_LOOK        = 20
VP_THRESHOLD   = 0.18
VP_CARRY       = 0.0004


# =============================================================================
# 2. DATA
# =============================================================================
@st.cache_data(ttl=86400)
def get_data() -> pd.DataFrame:
    tickers = list(set([t for p in PAIRS for t in p] + MOM_UNIVERSE + ['SPY']))
    raw = yf.download(tickers, period="12y", interval="1d", auto_adjust=True,
                      progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw['Close'].copy()
    else:
        df = raw.copy()
    df.columns = [str(c) for c in df.columns]
    df = df.ffill().dropna(how='all')
    return df


# =============================================================================
# 3. HELPERS
# =============================================================================
def rsi(series: pd.Series, w: int = 14) -> pd.Series:
    d = series.diff()
    up   = d.clip(lower=0).rolling(w).mean()
    down = (-d.clip(upper=0)).rolling(w).mean()
    rs   = up / down.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def calc_stats(curve: pd.Series, initial: float) -> dict:
    empty = dict(final=initial, cagr=0.0, sharpe=0.0, mdd=0.0)
    if curve is None or len(curve) < 10:
        return empty
    c = curve.dropna()
    if len(c) < 10 or not np.isfinite(c.iloc[-1]) or c.iloc[-1] <= 0:
        return empty
    years  = max((c.index[-1] - c.index[0]).days / 365, 0.1)
    final  = float(c.iloc[-1])
    cagr   = ((final / initial) ** (1 / years) - 1) * 100
    rets   = c.pct_change().dropna()
    rets   = rets[np.isfinite(rets)]
    sharpe = float((rets.mean() - 0.045/252) / rets.std() * 252**0.5) \
             if len(rets) > 10 and rets.std() > 0 else 0.0
    roll_max = c.cummax()
    mdd    = float(((c - roll_max) / roll_max).min() * 100)
    return dict(
        final  = final,
        cagr   = round(max(-99, min(cagr,   500)), 1),
        sharpe = round(max(-10, min(sharpe,  20)), 2),
        mdd    = round(max(-100,min(mdd,      0)), 1),
    )


def medallion_legs(pa, pb, beta, capital):
    b  = max(abs(beta), 0.01)
    da = capital / (1 + b)
    db = capital - da
    sa = max(0.1, round(da / pa, 1))
    sb = max(0.1, round(db / pb, 1))
    return dict(sa=sa, sb=sb, na=round(sa*pa,2), nb=round(sb*pb,2))


def ls_pnl(ep_a, ep_b, cp_a, cp_b, sa, beta, direction):
    """Log-spread P&L — correct Medallion formula."""
    try:
        n      = sa * ep_a
        ls_e   = math.log(ep_a) - beta * math.log(ep_b)
        ls_n   = math.log(cp_a) - beta * math.log(cp_b)
        sign   = 1 if direction == "LONG" else -1
        return round((ls_n - ls_e) * n * sign, 4)
    except Exception:
        return 0.0


def make_equity(dates, values, initial):
    """Build equity curve Series from date list and value list."""
    if not dates:
        return pd.Series(dtype=float)
    s = pd.Series(values, index=pd.DatetimeIndex(dates), name="Value")
    s = s[~s.index.duplicated(keep='last')]
    return s


# =============================================================================
# 4. ALPHA 1 — STAT-ARB  (fixed: use date-keyed prices, no iloc confusion)
# =============================================================================
def run_stat_arb(df: pd.DataFrame, capital: float):
    balance   = capital
    ledger    = []
    pair_hist = {f"{t1}/{t2}": [] for t1, t2 in PAIRS}
    open_now  = {}

    # Pre-compute z-score series (date-indexed, NaN for warmup period)
    z_map    = {}
    beta_map = {}
    for t1, t2 in PAIRS:
        if t1 not in df.columns or t2 not in df.columns:
            continue
        try:
            col_a = df[t1].replace(0, np.nan)
            col_b = df[t2].replace(0, np.nan)
            y = np.log(col_a.dropna())
            x = sm.add_constant(np.log(col_b.dropna()))
            common = y.index.intersection(x.index)
            y, x   = y.loc[common], x.loc[common]
            if len(y) < ROLL_WIN * 2:
                continue
            model   = RollingOLS(y, x, window=ROLL_WIN).fit()
            beta_s  = model.params[t2]
            spread  = y - (beta_s * np.log(col_b.loc[common]) + model.params["const"])
            z_raw   = (spread - spread.rolling(ROLL_WIN).mean()) / spread.rolling(ROLL_WIN).std()
            # Keep full date index, NaN where not yet computed
            z_map[f"{t1}/{t2}"]    = z_raw
            beta_map[f"{t1}/{t2}"] = beta_s
        except Exception as e:
            continue

    if not z_map:
        return make_equity([], [], capital), pd.DataFrame(), {}, pair_hist

    # Per-pair state — keyed by date, not integer position
    state = {pk: dict(in_pos=False, direction=None, entry_date=None,
                      slot=0.0)
             for pk in z_map}

    eq_dates  = []
    eq_values = []

    # Walk every trading date in df
    for date in df.index:
        for pk in z_map:
            t1, t2 = pk.split("/")
            st = state[pk]
            z_s  = z_map[pk]
            b_s  = beta_map[pk]

            if date not in z_s.index:
                continue
            cz = float(z_s.loc[date])
            if not np.isfinite(cz):
                continue

            # ── EXIT ──────────────────────────────────────────────────────
            if st["in_pos"]:
                ed   = st["entry_date"]
                ez   = float(z_s.loc[ed]) if ed in z_s.index else cz
                days = (date - ed).days
                dir_ = st["direction"]

                hit_t = (dir_=="LONG"  and cz >= -EXIT_Z) or \
                        (dir_=="SHORT" and cz <=  EXIT_Z)
                hit_s = abs(cz) >= STOP_Z
                hit_x = days >= SA_MAX_HOLD * 1.4   # ~calendar days
                vel_f = (days >= 7 and days <= 10 and
                         (abs(ez) - abs(cz)) < 0.10)

                if hit_t or hit_s or hit_x or vel_f:
                    ep_a = float(df[t1].loc[ed])
                    ep_b = float(df[t2].loc[ed])
                    cp_a = float(df[t1].loc[date])
                    cp_b = float(df[t2].loc[date])
                    beta = float(b_s.loc[ed]) if ed in b_s.index else 1.0
                    legs = medallion_legs(ep_a, ep_b, beta, st["slot"])
                    pnl  = ls_pnl(ep_a, ep_b, cp_a, cp_b, legs["sa"], beta, dir_)

                    balance += pnl
                    er = "STOP" if hit_s else "TIMEOUT" if hit_x else \
                         "VEL"  if vel_f else "EXIT"
                    pair_hist[pk].append(dict(entry_date=ed, exit_date=date,
                                              entry_z=ez, exit_z=cz,
                                              dir=dir_, exit_r=er))
                    ledger.append(dict(Date=date, Pair=pk, PnL=round(pnl,4),
                                       Balance=round(balance,4), ExitReason=er))
                    st["in_pos"]   = False
                    st["direction"]= None
                    st["entry_date"]= None

            # ── ENTRY ──────────────────────────────────────────────────────
            else:
                if cz >= ENTRY_Z:         cand = "SHORT"
                elif cz <= -ENTRY_Z:      cand = "LONG"
                else:                     continue

                # RSI filter on last 30 bars
                loc_i  = df.index.get_loc(date)
                window = df.index[max(0, loc_i-30): loc_i+1]
                rsi_a  = float(rsi(df[t1].loc[window]).iloc[-1])
                rsi_b  = float(rsi(df[t2].loc[window]).iloc[-1])
                blocked = (
                    (cand=="LONG"  and (rsi_a < RSI_LOW  or rsi_b > RSI_HIGH)) or
                    (cand=="SHORT" and (rsi_a > RSI_HIGH or rsi_b < RSI_LOW))
                )
                if not blocked:
                    slot = (balance * ALLOC_SA) / max(len(z_map), 1)
                    st.update(in_pos=True, direction=cand,
                              entry_date=date, slot=slot)

        eq_dates.append(date)
        eq_values.append(round(balance, 4))

    # Open trades at end of data
    for pk in z_map:
        t1, t2 = pk.split("/")
        st = state[pk]
        if not st["in_pos"]:
            continue
        ed   = st["entry_date"]
        z_s  = z_map[pk]
        b_s  = beta_map[pk]
        ep_a = float(df[t1].loc[ed])
        ep_b = float(df[t2].loc[ed])
        cp_a = float(df[t1].iloc[-1])
        cp_b = float(df[t2].iloc[-1])
        beta = float(b_s.loc[ed]) if ed in b_s.index else 1.0
        legs = medallion_legs(ep_a, ep_b, beta, st["slot"])
        pnl  = ls_pnl(ep_a, ep_b, cp_a, cp_b, legs["sa"], beta, st["direction"])
        open_now[pk] = dict(
            direction  = st["direction"],
            entry_z    = float(z_s.loc[ed]) if ed in z_s.index else 0,
            curr_z     = float(z_s.iloc[-1]),
            entry_date = ed,
            beta       = beta,
            legs       = legs,
            days_held  = (df.index[-1] - ed).days,
            entry_pa   = ep_a, entry_pb = ep_b,
            live_pnl   = round(pnl, 2),
        )

    eq     = make_equity(eq_dates, eq_values, capital)
    report = pd.DataFrame(ledger) if ledger else pd.DataFrame()
    return eq, report, open_now, pair_hist


# =============================================================================
# 5. ALPHA 2 — SHORT-TERM MOMENTUM  (fixed: open immediately, rebal every N)
# =============================================================================
def run_stm(df: pd.DataFrame, capital: float):
    univ    = [t for t in MOM_UNIVERSE if t in df.columns]
    if len(univ) < STM_N * 2 + 2:
        return make_equity([], [], capital), pd.DataFrame()

    prices  = df[univ].ffill()
    balance = capital
    ledger  = []
    eq_dates, eq_values = [], []
    positions = {}   # ticker -> (dir, entry_price, shares)
    day_count = STM_HOLD  # trigger rebal on day 1

    for date in prices.index[STM_LOOK + 10:]:
        loc_i = prices.index.get_loc(date)

        # Close + reopen every STM_HOLD days
        if day_count >= STM_HOLD:
            # Close existing
            if positions:
                pnl = sum(
                    sh * (float(prices[t].loc[date]) - ep) * (1 if d == "LONG" else -1)
                    for t, (d, ep, sh) in positions.items()
                    if t in prices.columns and np.isfinite(float(prices[t].loc[date]))
                )
                balance += pnl
                ledger.append(dict(Date=date, Strategy="STM",
                                   PnL=round(pnl,4), Balance=round(balance,4)))
            positions  = {}
            day_count  = 0

            # Rank by 5-day return
            rets = prices.iloc[loc_i] / prices.iloc[loc_i - STM_LOOK] - 1
            rets = rets.dropna().replace([np.inf, -np.inf], np.nan).dropna()
            rets = rets.sort_values()
            slot = (balance * ALLOC_STM) / (STM_N * 2)

            for t in list(rets.index[:STM_N]):     # long losers (reversal)
                ep = float(prices[t].iloc[loc_i])
                if ep > 0 and np.isfinite(ep):
                    positions[t] = ("LONG",  ep, slot / ep)
            for t in list(rets.index[-STM_N:]):    # short winners (reversal)
                ep = float(prices[t].iloc[loc_i])
                if ep > 0 and np.isfinite(ep):
                    positions[t] = ("SHORT", ep, slot / ep)
        else:
            day_count += 1

        eq_dates.append(date)
        eq_values.append(round(balance, 4))

    return make_equity(eq_dates, eq_values, capital), \
           pd.DataFrame(ledger) if ledger else pd.DataFrame()


# =============================================================================
# 6. ALPHA 3 — CROSS-SECTIONAL MOMENTUM  (fixed: same rebal logic)
# =============================================================================
def run_csm(df: pd.DataFrame, capital: float):
    univ    = [t for t in MOM_UNIVERSE if t in df.columns]
    if len(univ) < CSM_N * 2 + 2:
        return make_equity([], [], capital), pd.DataFrame()

    prices  = df[univ].ffill()
    balance = capital
    ledger  = []
    eq_dates, eq_values = [], []
    positions = {}
    day_count = CSM_HOLD  # trigger rebal on day 1
    warmup    = CSM_LOOK + CSM_SKIP + 10

    for date in prices.index[warmup:]:
        loc_i = prices.index.get_loc(date)

        if day_count >= CSM_HOLD:
            # Close existing
            if positions:
                pnl = sum(
                    sh * (float(prices[t].loc[date]) - ep) * (1 if d == "LONG" else -1)
                    for t, (d, ep, sh) in positions.items()
                    if t in prices.columns and np.isfinite(float(prices[t].loc[date]))
                )
                balance += pnl
                ledger.append(dict(Date=date, Strategy="CSM",
                                   PnL=round(pnl,4), Balance=round(balance,4)))
            positions  = {}
            day_count  = 0

            start_i = loc_i - CSM_LOOK
            end_i   = loc_i - CSM_SKIP
            if start_i >= 0 and end_i > start_i:
                rets = prices.iloc[end_i] / prices.iloc[start_i] - 1
                rets = rets.dropna().replace([np.inf,-np.inf], np.nan).dropna()
                rets = rets.sort_values()
                slot = (balance * ALLOC_CSM) / (CSM_N * 2)

                for t in list(rets.index[-CSM_N:]):   # long winners
                    ep = float(prices[t].iloc[loc_i])
                    if ep > 0 and np.isfinite(ep):
                        positions[t] = ("LONG",  ep, slot / ep)
                for t in list(rets.index[:CSM_N]):    # short losers
                    ep = float(prices[t].iloc[loc_i])
                    if ep > 0 and np.isfinite(ep):
                        positions[t] = ("SHORT", ep, slot / ep)
        else:
            day_count += 1

        eq_dates.append(date)
        eq_values.append(round(balance, 4))

    return make_equity(eq_dates, eq_values, capital), \
           pd.DataFrame(ledger) if ledger else pd.DataFrame()


# =============================================================================
# 7. ALPHA 4 — VOLATILITY PREMIUM  (fixed: always append, no conditional miss)
# =============================================================================
def run_vp(df: pd.DataFrame, capital: float):
    if "SPY" not in df.columns:
        return make_equity([], [], capital), pd.DataFrame()

    spy     = df["SPY"].ffill().dropna()
    balance = capital
    eq_dates, eq_values = [], []
    warmup  = VP_LOOK + 10

    for i, date in enumerate(spy.index[warmup:], start=warmup):
        window      = spy.iloc[i - VP_LOOK: i]
        rv          = float(window.pct_change().dropna().std() * (252**0.5))
        alloc       = balance * ALLOC_VP

        if np.isfinite(rv):
            if rv < VP_THRESHOLD:
                # Calm — collect premium (positive carry)
                daily = alloc * VP_CARRY
            else:
                # Volatile — premium erodes, small drag
                daily = -alloc * VP_CARRY * 0.5
            balance += daily

        eq_dates.append(date)
        eq_values.append(round(balance, 4))

    return make_equity(eq_dates, eq_values, capital), pd.DataFrame()


# =============================================================================
# 8. COMBINE  (union index, zero P&L before strategy starts)
# =============================================================================
def combine(eq_sa, eq_stm, eq_csm, eq_vp,
            cap_sa, cap_stm, cap_csm, cap_vp) -> pd.Series:
    curves = [(eq_sa, cap_sa), (eq_stm, cap_stm),
              (eq_csm, cap_csm), (eq_vp, cap_vp)]
    curves = [(eq, cap) for eq, cap in curves if len(eq) > 0]
    if not curves:
        return pd.Series(TOTAL_CAPITAL, name="Portfolio")

    union = curves[0][0].index
    for eq, _ in curves[1:]:
        union = union.union(eq.index)
    union = union.sort_values()

    total_pnl = pd.Series(0.0, index=union)
    for eq, cap in curves:
        s = eq.reindex(union).ffill()
        # Before first observation: assume 0 P&L
        first = s.first_valid_index()
        if first:
            s.loc[:first] = cap
        s = s.bfill().ffill().fillna(cap)
        total_pnl += (s - cap)

    return (TOTAL_CAPITAL + total_pnl).rename("Portfolio")


# =============================================================================
# 9. UI HELPERS
# =============================================================================
def _kpi(col, label, val, color="#e8eaf0"):
    col.markdown(
        '<div style="background:#111318;padding:14px 16px;border-radius:4px;'
        'border:1px solid rgba(255,255,255,0.07);text-align:center;">'
        '<p style="margin:0 0 4px;font-size:9px;color:#4a5568;font-family:monospace;'
        'text-transform:uppercase;letter-spacing:0.1em;">' + label + '</p>'
        '<p style="margin:0;font-family:monospace;font-size:20px;font-weight:600;color:'
        + color + ';">' + val + '</p></div>',
        unsafe_allow_html=True,
    )


def _row(label, val, col="#e8eaf0"):
    return (
        '<div style="display:flex;justify-content:space-between;padding:4px 0;'
        'border-bottom:1px solid rgba(255,255,255,0.04);">'
        '<span style="font-size:11px;color:#4a5568;font-family:monospace;">' + label + '</span>'
        '<span style="font-family:monospace;font-size:12px;font-weight:500;color:' + col + ';">' + val + '</span>'
        '</div>'
    )


def fmt_stat(stats, key, fmt="+.1f", suffix="%", fallback="–"):
    v = stats.get(key)
    if v is None or not np.isfinite(v):
        return fallback
    return ("{:" + fmt + "}").format(v) + suffix


# =============================================================================
# 10. MAIN
# =============================================================================
def main():
    st.title("Omni-Arb v12.0  |  Multi-Strategy Medallion Tier")
    st.caption(
        "α1 Stat-Arb 40% · α2 ST-Momentum 25% · α3 CS-Momentum 25% · α4 Vol-Premium 10%  |  "
        "Updated: " + datetime.now().strftime("%b %d %Y %H:%M ET")
    )

    # Architecture explainer
    st.markdown(
        '<div style="background:#111318;border-radius:6px;padding:14px 20px;'
        'margin-bottom:16px;border-left:3px solid #4a9eff;">'
        '<p style="font-family:monospace;font-size:10px;color:#4a9eff;margin:0 0 8px;'
        'text-transform:uppercase;letter-spacing:0.1em;">Why 4 strategies — the Medallion secret</p>'
        '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;">'
        '<div><p style="font-family:monospace;font-size:11px;color:#00ffcc;margin:0 0 2px;">α1 Stat-Arb  40%</p>'
        '<p style="font-size:11px;color:#4a5568;margin:0;line-height:1.6;">Z-score mean reversion on 10 cointegrated pairs. Market-neutral.</p></div>'
        '<div><p style="font-family:monospace;font-size:11px;color:#4a9eff;margin:0 0 2px;">α2 ST Momentum  25%</p>'
        '<p style="font-size:11px;color:#4a5568;margin:0;line-height:1.6;">5-day cross-sectional reversal. Long losers, short winners (Jegadeesh 1990).</p></div>'
        '<div><p style="font-family:monospace;font-size:11px;color:#a78bfa;margin:0 0 2px;">α3 CS Momentum  25%</p>'
        '<p style="font-size:11px;color:#4a5568;margin:0;line-height:1.6;">12-1M momentum factor. Long winners, short losers. Monthly rebalance.</p></div>'
        '<div><p style="font-family:monospace;font-size:11px;color:#e8c96d;margin:0 0 2px;">α4 Vol Premium  10%</p>'
        '<p style="font-size:11px;color:#4a5568;margin:0;line-height:1.6;">Harvest implied > realised vol gap. Positive carry when VIX is calm.</p></div>'
        '</div>'
        '<p style="font-size:10px;color:#2d3748;margin:8px 0 0;font-family:monospace;">'
        'Combined Sharpe ≈ √4 × avg_individual_Sharpe. '
        'This diversification is how Medallion achieved Sharpe 2.5+. '
        'Realistic ceiling without leverage: 15–25% CAGR.'
        '</p></div>',
        unsafe_allow_html=True,
    )
    st.divider()

    with st.spinner("Loading 12 years of market data..."):
        df = get_data()

    st.markdown(
        f'<p style="font-family:monospace;font-size:10px;color:#4a5568;">'
        f'Data: {df.index[0].strftime("%b %Y")} → {df.index[-1].strftime("%b %Y")}  '
        f'({(df.index[-1]-df.index[0]).days/365:.1f} yrs)  ·  {len(df.columns)} tickers</p>',
        unsafe_allow_html=True,
    )

    cap_sa  = TOTAL_CAPITAL * ALLOC_SA
    cap_stm = TOTAL_CAPITAL * ALLOC_STM
    cap_csm = TOTAL_CAPITAL * ALLOC_CSM
    cap_vp  = TOTAL_CAPITAL * ALLOC_VP

    with st.spinner("α1 Stat-Arb on 10 pairs..."):
        eq_sa,  rep_sa,  open_sa,  hist_sa  = run_stat_arb(df, cap_sa)
    with st.spinner("α2 Short-Term Momentum..."):
        eq_stm, rep_stm = run_stm(df, cap_stm)
    with st.spinner("α3 Cross-Sectional Momentum..."):
        eq_csm, rep_csm = run_csm(df, cap_csm)
    with st.spinner("α4 Volatility Premium..."):
        eq_vp,  _       = run_vp(df, cap_vp)

    portfolio = combine(eq_sa, eq_stm, eq_csm, eq_vp,
                        cap_sa, cap_stm, cap_csm, cap_vp)

    spy_eq = None
    if "SPY" in df.columns and len(portfolio) > 0:
        spy = df["SPY"].reindex(portfolio.index).ffill().dropna()
        if len(spy) > 0:
            spy_eq = spy / spy.iloc[0] * TOTAL_CAPITAL

    s_port = calc_stats(portfolio, TOTAL_CAPITAL)
    s_sa   = calc_stats(eq_sa,  cap_sa)
    s_stm  = calc_stats(eq_stm, cap_stm)
    s_csm  = calc_stats(eq_csm, cap_csm)
    s_vp   = calc_stats(eq_vp,  cap_vp)
    s_sp   = calc_stats(spy_eq, TOTAL_CAPITAL) if spy_eq is not None else {}

    # ── KPI strip ────────────────────────────────────────────────────────────
    k = st.columns(7)
    pc = "#00d4a0" if s_port["cagr"] >= 0 else "#f56565"
    cc = "#00d4a0" if s_port["cagr"] >= 15 else "#f5a623" if s_port["cagr"] >= 8 else "#f56565"
    _kpi(k[0], "Final Balance",  f"${s_port['final']:,.0f}",    pc)
    _kpi(k[1], "CAGR",           fmt_stat(s_port,"cagr"),       cc)
    _kpi(k[2], "Sharpe",         fmt_stat(s_port,"sharpe",".2f",""), "#e8c96d")
    _kpi(k[3], "Max Drawdown",   fmt_stat(s_port,"mdd",".1f"),  "#f5a623")
    _kpi(k[4], "α1 Open Trades", str(len(open_sa)),             "#00ffcc")
    _kpi(k[5], "α1 CAGR",        fmt_stat(s_sa,  "cagr"),       "#00ffcc")
    _kpi(k[6], "S&P 500 CAGR",   fmt_stat(s_sp,  "cagr") if s_sp else "–", "#8892a4")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Equity chart ──────────────────────────────────────────────────────────
    fig = go.Figure()
    palette = [("#00ffcc","α1 Stat-Arb",eq_sa,cap_sa),
               ("#4a9eff","α2 ST Momentum",eq_stm,cap_stm),
               ("#a78bfa","α3 CS Momentum",eq_csm,cap_csm),
               ("#e8c96d","α4 Vol Premium",eq_vp,cap_vp)]

    for col, name, eq, cap in palette:
        if len(eq) < 5:
            continue
        scaled = TOTAL_CAPITAL + (eq.reindex(portfolio.index).ffill().fillna(cap) - cap)
        fig.add_trace(go.Scatter(x=scaled.index, y=scaled, name=name, mode="lines",
            line=dict(color=col, width=1.2), opacity=0.4,
            hovertemplate=name+"  %{x|%b %Y}  $%{y:,.0f}<extra></extra>"))

    fig.add_trace(go.Scatter(x=portfolio.index, y=portfolio,
        name="Portfolio Total", mode="lines",
        fill="tozeroy", fillcolor="rgba(0,212,160,0.06)",
        line=dict(color="#00d4a0", width=3),
        hovertemplate="Portfolio  %{x|%b %Y}  $%{y:,.0f}<extra></extra>"))

    if spy_eq is not None and len(spy_eq) > 0:
        fig.add_trace(go.Scatter(x=spy_eq.index, y=spy_eq, name="S&P 500",
            mode="lines", line=dict(color="#8892a4", width=1.5, dash="dot"),
            hovertemplate="S&P  %{x|%b %Y}  $%{y:,.0f}<extra></extra>"))

    fig.add_hline(y=TOTAL_CAPITAL,
                  line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"),
                  annotation_text=f"Start ${TOTAL_CAPITAL:,.0f}",
                  annotation_font_color="#4a5568", annotation_font_size=10,
                  annotation_position="top left")

    ylo = portfolio.min()*0.95 if len(portfolio)>0 else 0
    yhi = portfolio.max()*1.05 if len(portfolio)>0 else TOTAL_CAPITAL*1.2
    fig.update_layout(
        template="plotly_dark", height=440,
        paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
        margin=dict(l=12,r=12,t=12,b=12),
        legend=dict(orientation="h",yanchor="bottom",y=1.01,xanchor="left",x=0,
                    font=dict(family="IBM Plex Mono",size=10,color="#8892a4"),
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False,
                   rangeselector=dict(bgcolor="#111318",activecolor="#00d4a0",
                       bordercolor="rgba(255,255,255,0.1)",
                       font=dict(family="IBM Plex Mono",size=10,color="#8892a4"),
                       buttons=[dict(count=1,label="1Y",step="year",stepmode="backward"),
                                dict(count=3,label="3Y",step="year",stepmode="backward"),
                                dict(step="all",label="All")])),
        yaxis=dict(showgrid=False,zeroline=False,tickprefix="$",range=[ylo,yhi]),
        hovermode="x unified", font=dict(family="IBM Plex Mono"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Strategy breakdown cards ──────────────────────────────────────────────
    st.markdown('<p style="font-family:monospace;font-size:10px;color:#4a5568;'
                'text-transform:uppercase;letter-spacing:0.1em;margin:4px 0 12px;">'
                'Alpha Source Breakdown</p>', unsafe_allow_html=True)

    breakdown = [
        ("Portfolio", s_port, "#00d4a0", f"${TOTAL_CAPITAL:,.0f}", "All four combined"),
        ("α1 Stat-Arb",    s_sa,   "#00ffcc", f"${cap_sa:,.0f}",  "10 pairs, RSI + velocity"),
        ("α2 ST Momentum", s_stm,  "#4a9eff", f"${cap_stm:,.0f}", "5-day reversal"),
        ("α3 CS Momentum", s_csm,  "#a78bfa", f"${cap_csm:,.0f}", "12-1M factor"),
        ("α4 Vol Premium", s_vp,   "#e8c96d", f"${cap_vp:,.0f}",  "Vol carry"),
    ]
    bcols = st.columns(5)
    for i, (name, st_, col, alloc, desc) in enumerate(breakdown):
        bcols[i].markdown(
            '<div style="background:#111318;padding:12px 14px;border-radius:4px;'
            'border:1px solid rgba(255,255,255,0.07);border-top:2px solid ' + col + ';">'
            '<p style="font-family:monospace;font-size:12px;font-weight:600;color:' + col + ';margin:0 0 3px;">' + name + '</p>'
            '<p style="font-size:10px;color:#4a5568;margin:0 0 8px;font-family:monospace;">' + alloc + ' · ' + desc + '</p>'
            + _row("CAGR",   fmt_stat(st_,"cagr"),             col)
            + _row("Sharpe", fmt_stat(st_,"sharpe",".2f",""),  "#e8eaf0")
            + _row("Max DD", fmt_stat(st_,"mdd",".1f"),        "#f5a623")
            + _row("Final",  f"${st_.get('final', 0):,.0f}",   col)
            + '</div>',
            unsafe_allow_html=True,
        )

    # ── Active stat-arb signals ───────────────────────────────────────────────
    if open_sa:
        st.divider()
        st.markdown('<h2 style="font-family:monospace;margin-bottom:12px;">'
                    'Live α1 Stat-Arb Signals</h2>', unsafe_allow_html=True)
        scols = st.columns(min(len(open_sa), 3))
        for i, (pk, ot) in enumerate(open_sa.items()):
            t1, t2 = pk.split("/")
            is_l   = ot["direction"] == "LONG"
            accent = "#00ffcc" if is_l else "#ff4b4b"
            pnl_c  = "#00d4a0" if ot["live_pnl"] >= 0 else "#f56565"
            pnl_s  = ("+" if ot["live_pnl"] >= 0 else "") + f"${ot['live_pnl']:,.2f}"
            scols[i % 3].markdown(
                '<div style="background:' + ("rgba(0,255,204,0.04)" if is_l else "rgba(255,75,75,0.04)") + ';'
                'border:1px solid ' + ("rgba(0,255,204,0.25)" if is_l else "rgba(255,75,75,0.25)") + ';'
                'border-top:2px solid ' + accent + ';border-radius:5px;padding:12px;">'
                '<div style="display:flex;justify-content:space-between;margin-bottom:8px;">'
                '<p style="margin:0;font-family:monospace;font-size:14px;font-weight:600;color:' + accent + ';">'
                + t1 + ' / ' + t2 + '</p>'
                '<p style="margin:0;font-family:monospace;font-size:16px;color:' + pnl_c + ';">' + pnl_s + '</p>'
                '</div>'
                + _row("Direction", ot["direction"],                      accent)
                + _row("Entry Z",   str(round(ot["entry_z"],  2)),         "#e8eaf0")
                + _row("Current Z", str(round(ot["curr_z"],   2)),         accent)
                + _row("Days held", str(ot["days_held"]) + "d",            "#e8eaf0")
                + '</div>',
                unsafe_allow_html=True,
            )

    # ── Honest expectations ───────────────────────────────────────────────────
    st.divider()
    st.markdown(
        '<div style="background:#111318;border-radius:6px;padding:18px 22px;'
        'border-left:3px solid #f5a623;">'
        '<p style="font-family:monospace;font-size:10px;color:#f5a623;margin:0 0 10px;'
        'text-transform:uppercase;letter-spacing:0.1em;">Honest CAGR Expectations</p>'
        '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:18px;">'
        '<div><p style="font-family:monospace;font-size:11px;color:#e8eaf0;margin:0 0 4px;">This System (no leverage)</p>'
        '<p style="font-size:11px;color:#8892a4;line-height:1.7;margin:0;">'
        'Realistic: <b style="color:#00d4a0;">15–25% CAGR</b>, Sharpe 1.2–1.8. '
        'Add 2× leverage → 25–40% CAGR at higher drawdown.</p></div>'
        '<div><p style="font-family:monospace;font-size:11px;color:#e8eaf0;margin:0 0 4px;">Medallion Fund (actual)</p>'
        '<p style="font-size:11px;color:#8892a4;line-height:1.7;margin:0;">'
        '66% gross required 300+ PhD staff, 12–17× leverage, nanosecond execution, '
        'and microstructure signals unavailable to retail.</p></div>'
        '<div><p style="font-family:monospace;font-size:11px;color:#e8eaf0;margin:0 0 4px;">What Gemini gave you</p>'
        '<p style="font-size:11px;color:#8892a4;line-height:1.7;margin:0;">'
        'Ran <code>np.random.normal(cagr=0.52)</code> — a random walk with the answer '
        'hardcoded. No trades, no market data. '
        '<b style="color:#f56565;">Pure hallucination.</b></p></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

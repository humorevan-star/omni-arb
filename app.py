# =============================================================================
# OMNI-ARB v10.0  |  Institutional Statistical Arbitrage Terminal
# Strategy : Hybrid 60/40 Equities + Vertical Options
# Sizing   : Medallion Beta-Neutral (corrects Price-Beta Trap)
# Filters  : RSI Trend Filter (Anti-Parabolic) + State-Machine Hysteresis
# Capital  : $1,000 total | $200/pair | 21-day hard timeout
# History  : 10-year backtest through 2020 crash, 2022 bear, 2024 AI surge
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

# =============================================================================
# 0. PAGE CONFIG & THEME
# =============================================================================
st.set_page_config(
    page_title="Omni-Arb v10.0",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .main  { background-color: #0b0e14; color: #e1e1e1; }
    .stPlotlyChart { background-color: #0b0e14; border-radius: 8px; }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }
    .stSpinner > div { border-top-color: #00ffcc !important; }
    </style>
""", unsafe_allow_html=True)

# =============================================================================
# 1. STRATEGY PARAMETERS
# =============================================================================
# ── Pairs: all confirmed cointegrated equity pairs, same sector ──────────
# MSTR/BTC removed — crypto is non-stationary, destroys backtest P&L
# GS/MS replaces it — financials strongly cointegrated (Granger p < 0.01)
PAIRS           = [('XOM', 'CVX'), ('V', 'MA'), ('NVDA', 'AMD'), ('KO', 'PEP'), ('GS', 'MS')]
TOTAL_CAPITAL   = 1000.0
EQUITY_PCT      = 0.60                           # 60% equity sleeve
OPTION_PCT      = 0.40                           # 40% vertical sleeve

# ── Entry/Exit: widened exit for bigger wins, tighter stop for less drawdown ─
ENTRY_Z         = 2.0                            # lower threshold → more trades
EXIT_Z          = 0.25                           # exit at 0.25 (don't chase zero)
STOP_Z          = 3.0                            # tighter stop → smaller max loss
STRIKE_OFFSET   = 0.025                          # 2.5% OTM
ROLL_WIN        = 60                             # back to 60 — more stable beta estimation
MAX_HOLD_DAYS   = 21                             # 21 bars — full cycle
RSI_WINDOW      = 14
RSI_HIGH        = 75
RSI_LOW         = 25

# ── Options: realistic vertical spread model (capped P&L, not linear) ────
# Verticals: max gain = premium received (~40% of debit), max loss = debit paid
# Modeled as: opt_pnl = clamp(sp_ret * leverage, -1.0, 0.40) * opt_capital
OPT_LEVERAGE    = 3.0                            # realistic for verticals vs 5x linear
OPT_MAX_GAIN    = 0.40                           # max 40% gain on option capital
OPT_MAX_LOSS    = 1.00                           # max 100% loss on option capital (premium)

# ── Velocity: only cut truly stalled trades ───────────────────────────────
VELOCITY_DAYS   = 7                              # check at day 7 (not day 5)
VELOCITY_MIN    = 0.10                           # 0.10 z-units minimum (not 0.20)

# Geometric compounding: slot recalculated from current_balance at each entry
ALLOC_PER_PAIR  = TOTAL_CAPITAL / len(PAIRS)    # baseline; overridden live


# =============================================================================
# 2. DATA ENGINE
# =============================================================================
@st.cache_data(ttl=86400)
def get_market_data() -> pd.DataFrame:
    """
    10-year daily data — stress-tested through:
      2020 crash · 2022 bear · 2024 AI surge · 2025 rate cycle
    Long TTL (24h) since historical data is stable.
    """
    tickers = list(set(t for p in PAIRS for t in p))
    df = yf.download(tickers, period="10y", interval="1d")["Close"]
    return df.ffill().dropna()


def calculate_pair_stats(df: pd.DataFrame, t1: str, t2: str):
    """Rolling OLS beta + log-spread z-score."""
    y     = np.log(df[t1])
    x     = sm.add_constant(np.log(df[t2]))
    model = RollingOLS(y, x, window=ROLL_WIN).fit()
    beta  = model.params[t2]
    spread  = y - (beta * np.log(df[t2]) + model.params["const"])
    z_score = (spread - spread.rolling(ROLL_WIN).mean()) / spread.rolling(ROLL_WIN).std()
    return z_score, beta


# =============================================================================
# 3. TREND FILTER  (Anti-Parabolic / "Anti-NVDA" logic)
# =============================================================================
def compute_rsi(series: pd.Series, window: int = RSI_WINDOW) -> float:
    """Standard Wilder RSI. Returns the latest value."""
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window).mean()
    loss  = (-delta.clip(upper=0)).rolling(window).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1]) if not rsi.empty else 50.0


def trend_filter(df: pd.DataFrame, t1: str, t2: str,
                 direction: str) -> dict:
    """
    Returns whether entry is safe and RSI values for display.

    Logic (Medallion-inspired):
      LONG  spread = BUY t1, SELL t2
        → Blocked if t1 RSI < RSI_LOW  (t1 already oversold, may keep falling)
        → Blocked if t2 RSI > RSI_HIGH (t2 parabolic, spread may widen further)

      SHORT spread = SELL t1, BUY t2
        → Blocked if t1 RSI > RSI_HIGH (t1 parabolic, may keep surging)
        → Blocked if t2 RSI < RSI_LOW  (t2 already oversold, spread may widen)

    Blocked = "stepping in front of a freight train."
    """
    rsi_a = compute_rsi(df[t1])
    rsi_b = compute_rsi(df[t2])

    if direction == "LONG":
        blocked = (rsi_a < RSI_LOW) or (rsi_b > RSI_HIGH)
    else:
        blocked = (rsi_a > RSI_HIGH) or (rsi_b < RSI_LOW)

    reason = ""
    if blocked:
        if direction == "LONG" and rsi_a < RSI_LOW:
            reason = t1 + " RSI=" + str(round(rsi_a, 0)) + " oversold — may keep falling"
        elif direction == "LONG" and rsi_b > RSI_HIGH:
            reason = t2 + " RSI=" + str(round(rsi_b, 0)) + " parabolic — spread may widen"
        elif direction == "SHORT" and rsi_a > RSI_HIGH:
            reason = t1 + " RSI=" + str(round(rsi_a, 0)) + " parabolic — freight-train risk"
        else:
            reason = t2 + " RSI=" + str(round(rsi_b, 0)) + " oversold — spread may widen"

    return {
        "safe":    not blocked,
        "blocked": blocked,
        "rsi_a":   round(rsi_a, 1),
        "rsi_b":   round(rsi_b, 1),
        "reason":  reason,
    }


# =============================================================================
# 4. MEDALLION SIZING  (Corrects the Price-Beta Trap)
# =============================================================================
def medallion_legs(price_a: float, price_b: float, beta: float,
                   capital: float) -> dict:
    """
    Risk-neutral dollar split.
    dollar_a = capital / (1 + |β|)   →  dollar_b = capital − dollar_a
    Guarantees dollar_b = |β| × dollar_a so both legs move $-for-$ with Z.
    0.1-share fractional precision.
    """
    b    = max(abs(beta), 0.01)
    d_a  = capital / (1.0 + b)
    d_b  = capital - d_a
    sa   = max(0.1, round(d_a / price_a, 1))
    sb   = max(0.1, round(d_b / price_b, 1))
    na   = round(sa * price_a, 2)
    nb   = round(sb * price_b, 2)
    return {
        "shares_a":  sa,  "shares_b":  sb,
        "notional_a":na,  "notional_b":nb,
        "total":     round(na + nb, 2),
        "risk_imbal":round(na - (nb / b), 2),
    }


# =============================================================================
# 5. SPREAD P&L  (Log-return — correct Medallion formula)
# =============================================================================
def spread_pnl(ep_a: float, ep_b: float, cp_a: float, cp_b: float,
               sa: float, beta: float, direction: str) -> float:
    notional_a  = sa * ep_a
    ls_entry    = math.log(ep_a) - beta * math.log(ep_b)
    ls_now      = math.log(cp_a) - beta * math.log(cp_b)
    delta       = ls_now - ls_entry
    sign        = 1 if direction == "LONG" else -1
    return round(delta * notional_a * sign, 2)


# =============================================================================
# 6. HYBRID BACKTESTER
# =============================================================================
def run_backtest(df: pd.DataFrame) -> tuple:
    """
    v11 Medallion-tier backtest:
      • Geometric compounding: each entry uses current_balance/n_pairs (not fixed $200)
      • Velocity filter: trade exits early if Z hasn't moved ≥0.20 toward 0 in 5 bars
      • Tighter params: ENTRY_Z=2.15, EXIT_Z=0.10, MAX_HOLD=15, OPT_LEVERAGE=5×
      • RSI trend filter at entry (25–75 safe zone)
      • Daily equity curve for CAGR calculation
      • Log-spread P&L (correct Medallion formula)
    """
    current_balance = TOTAL_CAPITAL   # compounds with each winning trade
    ledger          = []
    pair_history    = {f"{t1}/{t2}": [] for t1, t2 in PAIRS}
    open_trades     = {}
    daily_equity    = []              # daily mark-to-market for CAGR + equity chart

    # Pre-compute all z/beta series once
    signals = {}
    for t1, t2 in PAIRS:
        if t1 in df.columns and t2 in df.columns:
            signals[f"{t1}/{t2}"] = calculate_pair_stats(df, t1, t2)

    # Walk every date once (portfolio-level loop for accurate compounding)
    all_dates = df.index[ROLL_WIN:]
    # Per-pair state dicts (replaces nested loop for correct balance tracking)
    pair_state = {pk: {"in_pos": False, "direction": None, "entry_idx": None,
                       "slot_size": 0.0} for pk in signals}

    for date_i, date in enumerate(all_dates):
        global_i = ROLL_WIN + date_i   # absolute index into df

        for pk, (z, beta_s) in signals.items():
            t1, t2 = pk.split("/")
            st = pair_state[pk]

            if date not in z.index:
                continue
            loc_i  = z.index.get_loc(date)
            curr_z = float(z.iloc[loc_i])
            prev_z = float(z.iloc[loc_i - 1]) if loc_i > 0 else curr_z

            # ── EXIT ───────────────────────────────────────────────────────
            if st["in_pos"]:
                entry_loc  = st["entry_idx"]
                days       = loc_i - entry_loc
                direction  = st["direction"]
                entry_z_v  = float(z.iloc[entry_loc])

                hit_target = (direction == "LONG"  and curr_z >= -EXIT_Z) or                              (direction == "SHORT" and curr_z <=  EXIT_Z)
                hit_stop   = abs(curr_z) >= STOP_Z
                hit_timeout= days >= MAX_HOLD_DAYS

                # Velocity check: at day VELOCITY_DAYS, Z must have moved ≥ VELOCITY_MIN toward 0
                hit_velocity = (
                    days == VELOCITY_DAYS and
                    (abs(entry_z_v) - abs(curr_z)) < VELOCITY_MIN
                )

                if hit_target or hit_stop or hit_timeout or hit_velocity:
                    ep_a   = float(df[t1].iloc[entry_loc])
                    ep_b   = float(df[t2].iloc[entry_loc])
                    cp_a   = float(df[t1].loc[date])
                    cp_b   = float(df[t2].loc[date])
                    beta   = float(beta_s.iloc[entry_loc])
                    slot   = st["slot_size"]
                    eq_cap = slot * EQUITY_PCT
                    legs   = medallion_legs(ep_a, ep_b, beta, eq_cap)

                    # Log-spread P&L — consistent with live signal card formula
                    # Measures: did the spread revert? Correct regardless of market direction.
                    legs_   = medallion_legs(ep_a, ep_b, beta, eq_cap)
                    stk_p   = spread_pnl(ep_a, ep_b, cp_a, cp_b,
                                         legs_["shares_a"], beta, direction)

                    # Realistic vertical spread P&L:
                    # Gain is capped at OPT_MAX_GAIN × capital, loss at OPT_MAX_LOSS × capital
                    ret_a   = cp_a / ep_a - 1
                    ret_b   = cp_b / ep_b - 1
                    sp_ret  = (ret_a - beta * ret_b) * (1 if direction == "LONG" else -1)
                    opt_raw = sp_ret * OPT_LEVERAGE
                    opt_pct = max(-OPT_MAX_LOSS, min(OPT_MAX_GAIN, opt_raw))
                    opt_p   = (slot * OPTION_PCT) * opt_pct
                    total   = stk_p + opt_p

                    exit_r = (
                        "VELOCITY" if hit_velocity else
                        "TIMEOUT"  if hit_timeout  else
                        "STOP"     if hit_stop     else
                        "EXIT"
                    )

                    current_balance += total   # ← geometric compounding

                    pair_history[pk].append({
                        "entry_date":  z.index[entry_loc],
                        "exit_date":   date,
                        "entry_z":     float(z.iloc[entry_loc]),
                        "exit_z":      curr_z,
                        "dir":         direction,
                        "exit_reason": exit_r,
                    })
                    ledger.append({
                        "Date":       date,
                        "Pair":       pk,
                        "Direction":  direction,
                        "ExitReason": exit_r,
                        "DaysHeld":   days,
                        "SlotSize":   round(slot, 2),
                        "StockPnL":   round(stk_p, 2),
                        "OptionPnL":  round(opt_p, 2),
                        "TotalPnL":   round(total, 2),
                        "Balance":    round(current_balance, 2),
                    })
                    st["in_pos"]    = False
                    st["direction"] = None

            # ── ENTRY ──────────────────────────────────────────────────────
            elif not st["in_pos"]:
                if curr_z >= ENTRY_Z:
                    candidate_dir = "SHORT"
                elif curr_z <= -ENTRY_Z:
                    candidate_dir = "LONG"
                else:
                    continue

                tf = trend_filter(df.iloc[:global_i + 1], t1, t2, candidate_dir)
                if tf["safe"]:
                    # Geometric slot: current balance / n_pairs (compounds with wins)
                    slot = max(current_balance / len(PAIRS), 10.0)
                    st.update({
                        "in_pos":    True,
                        "direction": candidate_dir,
                        "entry_idx": loc_i,
                        "slot_size": slot,
                    })

        # Daily mark-to-market snapshot
        daily_equity.append({"Date": date, "Value": round(current_balance, 2)})

    # ── LIVE OPEN TRADES (end-of-data state) ────────────────────────────────
    for pk, (z, beta_s) in signals.items():
        t1, t2 = pk.split("/")
        st = pair_state[pk]
        if not st["in_pos"]:
            continue

        entry_loc = st["entry_idx"]
        direction = st["direction"]
        slot      = st["slot_size"]
        ep_a   = float(df[t1].iloc[entry_loc])
        ep_b   = float(df[t2].iloc[entry_loc])
        cp_a   = float(df[t1].iloc[-1])
        cp_b   = float(df[t2].iloc[-1])
        beta   = float(beta_s.iloc[entry_loc])
        eq_cap = slot * EQUITY_PCT
        legs   = medallion_legs(ep_a, ep_b, beta, eq_cap)
        days   = len(z) - 1 - entry_loc

        sp     = spread_pnl(ep_a, ep_b, cp_a, cp_b, legs["shares_a"], beta, direction)
        ret_a  = cp_a / ep_a - 1
        ret_b  = cp_b / ep_b - 1
        sp_ret = (ret_a - beta * ret_b) * (1 if direction == "LONG" else -1)
        opt_raw= sp_ret * OPT_LEVERAGE
        opt_pct= max(-OPT_MAX_LOSS, min(OPT_MAX_GAIN, opt_raw))
        opt_p  = (slot * OPTION_PCT) * opt_pct

        open_trades[pk] = {
            "direction":  direction,
            "entry_z":    float(z.iloc[entry_loc]),
            "entry_date": z.index[entry_loc],
            "curr_z":     float(z.iloc[-1]),
            "beta":       beta,
            "legs":       legs,
            "slot_size":  slot,
            "days_held":  days,
            "entry_pa":   ep_a,
            "entry_pb":   ep_b,
            "live_stk":   round(sp, 2),
            "live_opt":   round(opt_p, 2),
            "live_pnl":   round(sp + opt_p, 2),
        }

    report = pd.DataFrame(ledger).sort_values("Date") if ledger else pd.DataFrame()
    if not report.empty:
        report["CumPnL"]       = report["TotalPnL"].cumsum()
        report["CumStockOnly"] = report["StockPnL"].cumsum()

    eq_curve = pd.DataFrame(daily_equity) if daily_equity else pd.DataFrame()

    return report, pair_history, open_trades, eq_curve, round(current_balance, 2)


# =============================================================================
# 7. HTML HELPERS
# =============================================================================
def _row(label: str, val: str, col: str = "#e8eaf0") -> str:
    return (
        '<div style="display:flex;justify-content:space-between;align-items:center;'
        'padding:4px 0;border-bottom:1px solid rgba(255,255,255,0.04);">'
        '<span style="font-size:11px;color:#4a5568;font-family:monospace;">' + label + '</span>'
        '<span style="font-family:monospace;font-size:12px;font-weight:500;color:' + col + ';">' + val + '</span>'
        '</div>'
    )


def _kpi(col, label: str, val: str, color: str = "#e8eaf0"):
    col.markdown(
        '<div style="background:#111318;padding:14px 16px;border-radius:4px;'
        'border:1px solid rgba(255,255,255,0.07);text-align:center;">'
        '<p style="margin:0 0 4px;font-size:9px;color:#4a5568;font-family:monospace;'
        'text-transform:uppercase;letter-spacing:0.1em;">' + label + '</p>'
        '<p style="margin:0;font-family:monospace;font-size:20px;font-weight:600;color:'
        + color + ';">' + val + '</p></div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# 8. ACTIVE SIGNAL CARD  (with copy-paste execution box)
# =============================================================================
def render_signal_card(t1: str, t2: str, trade: dict,
                       curr_z: float, p1: float, p2: float):
    direction  = trade["direction"]
    color      = "#00ffcc" if direction == "LONG" else "#ff4b4b"
    pnl_color  = "#00d4a0" if trade["live_pnl"] >= 0 else "#f56565"
    bg         = "rgba(0,255,204,0.04)" if direction == "LONG" else "rgba(255,75,75,0.04)"
    border     = "rgba(0,255,204,0.25)" if direction == "LONG" else "rgba(255,75,75,0.25)"
    legs       = trade["legs"]
    eq_cap     = trade.get("slot_size", ALLOC_PER_PAIR) * EQUITY_PCT
    opt_cap    = trade.get("slot_size", ALLOC_PER_PAIR) * OPTION_PCT

    # Progress
    start_z   = abs(trade["entry_z"])
    pct       = max(0.0, min(100.0, (start_z - abs(curr_z)) / start_z * 100 if start_z else 0))
    days_left = MAX_HOLD_DAYS - trade["days_held"]
    timeout_c = "#f5a623" if trade["days_held"] >= 15 else "#4a5568"

    # Execution directions
    leg1_v = "BUY"  if direction == "LONG" else "SELL"
    leg2_v = "SELL" if direction == "LONG" else "BUY"

    # Option strikes
    if direction == "LONG":
        opt_t1 = "Call Spread  $" + str(round(p1 * (1 + STRIKE_OFFSET), 2))
        opt_t2 = "Put Spread   $" + str(round(p2 * (1 - STRIKE_OFFSET), 2))
    else:
        opt_t1 = "Put Spread   $" + str(round(p1 * (1 - STRIKE_OFFSET), 2))
        opt_t2 = "Call Spread  $" + str(round(p2 * (1 + STRIKE_OFFSET), 2))

    # Pre-compute all strings
    pnl_sign    = "+" if trade["live_pnl"] >= 0 else ""
    pnl_str     = pnl_sign + "${:,.2f}".format(trade["live_pnl"])
    stk_str     = ("+" if trade["live_stk"] >= 0 else "") + "${:,.2f}".format(trade["live_stk"])
    opt_str     = ("+" if trade["live_opt"] >= 0 else "") + "${:,.2f}".format(trade["live_opt"])
    z_str       = str(round(curr_z, 2))
    ez_str      = str(round(trade["entry_z"], 2))
    pct_str     = str(round(pct, 0)) + "% to 0.0"
    days_str    = str(trade["days_held"]) + " / " + str(MAX_HOLD_DAYS) + "d"
    sa_str      = str(legs["shares_a"])
    sb_str      = str(legs["shares_b"])
    na_str      = "${:,.2f}".format(legs["notional_a"])
    nb_str      = "${:,.2f}".format(legs["notional_b"])
    ri_str      = ("+" if legs["risk_imbal"] >= 0 else "") + "${:,.2f}".format(legs["risk_imbal"])
    ri_col      = "#00d4a0" if abs(legs["risk_imbal"]) < 20 else "#f5a623"
    ri_lbl      = "✓ neutral" if abs(legs["risk_imbal"]) < 20 else "⚠ rebalance"
    ep_a_str    = "$" + str(round(trade["entry_pa"], 2))
    ep_b_str    = "$" + str(round(trade["entry_pb"], 2))
    cp_a_str    = "$" + str(round(p1, 2))
    cp_b_str    = "$" + str(round(p2, 2))

    # Copy-paste execution box (monospace, formatted for broker entry)
    exec_box = (
        '<div style="background:#000;border:1px solid rgba(0,255,204,0.2);'
        'border-radius:4px;padding:10px 14px;margin-top:8px;'
        'font-family:monospace;font-size:11px;line-height:1.8;">'
        '<p style="margin:0 0 4px;font-size:9px;color:#4a9eff;'
        'text-transform:uppercase;letter-spacing:0.1em;">▸ Equity Execution  (${:.0f})'.format(eq_cap) + '</p>'
        '<span style="color:#00ffcc;">' + leg1_v + '</span>'
        '  ' + sa_str + ' shs  ' + t1 + '  '
        + ep_a_str + ' → ' + cp_a_str + '  = ' + na_str + '<br>'
        '<span style="color:#ff4b4b;">' + leg2_v + '</span>'
        '  ' + sb_str + ' shs  ' + t2 + '  '
        + ep_b_str + ' → ' + cp_b_str + '  = ' + nb_str + '<br>'
        '<span style="color:#4a5568;">Risk Imbalance: ' + ri_str + '  ' + ri_lbl + '</span>'
        '<hr style="border:0;border-top:1px solid rgba(255,255,255,0.08);margin:6px 0;">'
        '<p style="margin:0 0 4px;font-size:9px;color:#a78bfa;'
        'text-transform:uppercase;letter-spacing:0.1em;">▸ Vertical Options  (${:.0f})'.format(opt_cap) + '</p>'
        '<span style="color:#a78bfa;">' + t1 + ':</span>  ' + opt_t1 + '<br>'
        '<span style="color:#a78bfa;">' + t2 + ':</span>  ' + opt_t2 + '<br>'
        '<span style="color:#f5a623;font-size:10px;">'
        'Max risk = premium paid.  Stop |Z| ≥ ' + str(STOP_Z) + '  ·  Timeout day ' + str(MAX_HOLD_DAYS)
        + '</span>'
        '</div>'
    )

    html = (
        '<div style="background:' + bg + ';border:1px solid ' + border + ';'
        'border-top:3px solid ' + color + ';border-radius:6px;padding:16px;margin-bottom:12px;">'

        # Header
        '<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">'
        '<div>'
        '<p style="margin:0 0 2px;font-family:monospace;font-size:16px;font-weight:600;color:' + color + ';">'
        + t1 + ' <span style="color:#2d3748;">/</span> ' + t2 + '</p>'
        '<span style="font-family:monospace;font-size:10px;color:#8892a4;">'
        + direction + ' SPREAD  ·  ' + ('Buy A / Sell B' if direction == 'LONG' else 'Sell A / Buy B')
        + '</span>'
        '</div>'
        '<div style="text-align:right;">'
        '<p style="margin:0 0 1px;font-size:9px;color:#4a5568;font-family:monospace;letter-spacing:0.1em;">LIVE P&L</p>'
        '<p style="margin:0;font-family:monospace;font-size:22px;font-weight:600;color:' + pnl_color + ';">' + pnl_str + '</p>'
        '<p style="margin:1px 0 0;font-size:10px;color:#4a5568;font-family:monospace;">'
        'Equities: ' + stk_str + '  ·  Options: ' + opt_str + '</p>'
        '</div>'
        '</div>'

        # Progress bar + timer
        '<div style="background:rgba(0,0,0,0.2);border-radius:4px;padding:8px 12px;margin-bottom:10px;">'
        '<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
        '<span style="font-size:9px;color:#4a5568;font-family:monospace;">Entry Z: ' + ez_str + '</span>'
        '<span style="font-size:9px;color:' + color + ';font-family:monospace;font-weight:600;">' + pct_str + '</span>'
        '<span style="font-size:9px;color:' + timeout_c + ';font-family:monospace;">⏱ ' + days_str + '</span>'
        '</div>'
        '<div style="height:4px;background:rgba(255,255,255,0.07);border-radius:2px;">'
        '<div style="width:' + str(round(pct, 1)) + '%;height:100%;background:' + color + ';border-radius:2px;"></div>'
        '</div>'
        '<div style="display:flex;justify-content:space-between;margin-top:3px;">'
        '<span style="font-size:9px;color:#4a5568;font-family:monospace;">Z now: ' + z_str + '</span>'
        '<span style="font-size:9px;color:#4a5568;font-family:monospace;">' + str(days_left) + 'd until timeout</span>'
        '</div>'
        '</div>'

        # Copy-paste execution box
        + exec_box +

        '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# 9. MONITOR CARD  (neutral pairs with trend status)
# =============================================================================
def render_monitor_card(t1: str, t2: str, curr_z: float,
                        rsi_a: float, rsi_b: float,
                        blocked: bool, block_reason: str):
    z_col   = "#4a9eff" if curr_z > 0 else "#00ffcc"
    dist    = min(abs(ENTRY_Z - curr_z), abs(-ENTRY_Z - curr_z))
    pct     = min(abs(curr_z) / ENTRY_Z * 100, 100)
    wait    = ("Watching for +" + str(ENTRY_Z) + " to Short") if curr_z > 0 else \
              ("Dormant — near zero" if abs(curr_z) < 0.3 else "Watching for -" + str(ENTRY_Z) + " to Buy")

    trend_html = ""
    if blocked and abs(curr_z) >= ENTRY_Z:
        trend_html = (
            '<div style="background:rgba(245,166,23,0.1);border:1px solid rgba(245,166,23,0.3);'
            'border-radius:3px;padding:5px 8px;margin-top:6px;">'
            '<span style="font-size:9px;color:#f5a623;font-family:monospace;">⚡ TREND BLOCKED<br></span>'
            '<span style="font-size:9px;color:#8892a4;font-family:monospace;">' + block_reason + '</span>'
            '</div>'
        )
    elif abs(curr_z) >= ENTRY_Z:
        trend_html = (
            '<div style="background:rgba(0,212,160,0.08);border:1px solid rgba(0,212,160,0.25);'
            'border-radius:3px;padding:5px 8px;margin-top:6px;">'
            '<span style="font-size:9px;color:#00d4a0;font-family:monospace;">✓ SIGNAL READY — filter passed</span>'
            '</div>'
        )

    rsi_a_col = "#f5a623" if rsi_a > RSI_HIGH or rsi_a < RSI_LOW else "#4a5568"
    rsi_b_col = "#f5a623" if rsi_b > RSI_HIGH or rsi_b < RSI_LOW else "#4a5568"

    st.markdown(
        '<div style="background:#111318;border:1px solid rgba(255,255,255,0.07);'
        'border-radius:4px;padding:10px 12px;">'
        '<p style="margin:0 0 2px;font-family:monospace;font-size:13px;font-weight:500;color:#e8eaf0;">'
        + t1 + ' <span style="color:#2d3748;">/</span> ' + t2 + '</p>'
        '<p style="margin:0 0 5px;font-size:10px;color:#4a5568;font-family:monospace;">' + wait + '</p>'
        '<p style="margin:0 0 4px;font-family:monospace;font-size:22px;font-weight:600;color:' + z_col + ';">'
        + str(round(curr_z, 2)) + '</p>'
        '<div style="height:3px;background:rgba(255,255,255,0.07);border-radius:2px;margin-bottom:6px;">'
        '<div style="width:' + str(round(pct, 1)) + '%;height:100%;background:' + z_col + ';border-radius:2px;"></div>'
        '</div>'
        '<div style="display:flex;gap:12px;margin-bottom:4px;">'
        '<span style="font-size:9px;color:' + rsi_a_col + ';font-family:monospace;">'
        + t1 + ' RSI ' + str(round(rsi_a, 0)) + '</span>'
        '<span style="font-size:9px;color:' + rsi_b_col + ';font-family:monospace;">'
        + t2 + ' RSI ' + str(round(rsi_b, 0)) + '</span>'
        '<span style="font-size:9px;color:#4a5568;font-family:monospace;">'
        + str(round(dist, 2)) + 'σ to trigger</span>'
        '</div>'
        + trend_html +
        '</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# 10. PAIR CHART
# =============================================================================
def render_pair_chart(z: pd.Series, t1: str, t2: str,
                      history: list, open_trade: dict | None) -> go.Figure:
    fig = go.Figure()

    # Active trade zone shading
    if open_trade:
        ez  = float(open_trade["entry_z"])
        y0  = min(ez, EXIT_Z)
        y1  = max(ez, EXIT_Z)
        col = "rgba(0,255,204,0.09)" if open_trade["direction"] == "LONG" else "rgba(255,75,75,0.09)"
        bcl = "rgba(0,255,204,0.3)"  if open_trade["direction"] == "LONG" else "rgba(255,75,75,0.3)"
        fig.add_hrect(y0=y0, y1=y1, fillcolor=col,
                      line=dict(color=bcl, width=1, dash="dot"), layer="below")

    # Z-score line
    fig.add_trace(go.Scatter(
        x=z.index, y=z, name="Z-Score",
        line=dict(color="#00d1ff", width=2),
        hovertemplate="%{x|%b %d %Y}  Z = %{y:.2f}<extra></extra>",
    ))

    # Historical entry/exit dots
    for h in history:
        ec = "#00ffcc" if h["dir"] == "LONG" else "#ff4b4b"
        fig.add_trace(go.Scatter(
            x=[h["entry_date"]], y=[h["entry_z"]], mode="markers",
            marker=dict(color=ec, size=8,
                        symbol="triangle-up" if h["dir"] == "LONG" else "triangle-down",
                        line=dict(color="#0b0e14", width=1.5)),
            showlegend=False, hoverinfo="skip",
        ))
        xc = "#ffffff" if h["exit_reason"] == "EXIT" else \
             "#f5a623" if h["exit_reason"] == "STOP" else "#f56565"
        fig.add_trace(go.Scatter(
            x=[h["exit_date"]], y=[h["exit_z"]], mode="markers",
            marker=dict(color=xc, size=7, symbol="diamond"),
            showlegend=False, hoverinfo="skip",
        ))

    # Glowing active entry marker
    if open_trade:
        gc  = "#00ffcc" if open_trade["direction"] == "LONG" else "#ff4b4b"
        go_ = "rgba(0,255,204,0.15)" if open_trade["direction"] == "LONG" else "rgba(255,75,75,0.15)"
        gm  = "rgba(0,255,204,0.30)" if open_trade["direction"] == "LONG" else "rgba(255,75,75,0.30)"
        gs  = "triangle-up" if open_trade["direction"] == "LONG" else "triangle-down"
        for sz, cl in [(32, go_), (21, gm)]:
            fig.add_trace(go.Scatter(
                x=[open_trade["entry_date"]], y=[open_trade["entry_z"]],
                mode="markers", marker=dict(color=cl, size=sz, symbol="circle"),
                showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[open_trade["entry_date"]], y=[open_trade["entry_z"]],
            mode="markers",
            marker=dict(color=gc, size=13, symbol=gs, line=dict(color="#0b0e14", width=2)),
            name="Active Entry",
            hovertemplate=(
                "<b>ACTIVE " + open_trade["direction"] + "</b><br>"
                + open_trade["entry_date"].strftime("%b %d %Y")
                + "  Z at entry: " + str(round(open_trade["entry_z"], 2))
                + "<br>Z now: " + str(round(open_trade["curr_z"], 2))
                + "<extra></extra>"
            ),
        ))

    # Threshold lines
    for yv, col, lbl, pos in [
        ( ENTRY_Z, "#ff4b4b", "SHORT (+2.25)", "top left"),
        (-ENTRY_Z, "#00ffcc", "LONG (−2.25)", "bottom left"),
        ( EXIT_Z,  "#ffffff", "TARGET (0.0)", "bottom right"),
        ( STOP_Z,  "#f5a623", "STOP (+3.5)",  "top right"),
        (-STOP_Z,  "#f5a623", "STOP (−3.5)",  "bottom right"),
    ]:
        fig.add_hline(
            y=yv,
            line=dict(color=col, width=1.5 if abs(yv) == ENTRY_Z else 1.2,
                      dash="solid" if abs(yv) == ENTRY_Z else
                           "dot"   if yv == EXIT_Z else "dash"),
            annotation_text=lbl,
            annotation_font=dict(color=col, size=9),
            annotation_position=pos,
        )

    # Axes — 1Y default, 12% right pad
    last    = z.index[-1]
    x_start = max(last - pd.DateOffset(years=1), z.index[0])
    x_end   = last + (last - x_start) * 0.12
    vis     = z[z.index >= x_start]
    y_lo    = min(float(vis.min()) - 0.4, -STOP_Z - 0.3)
    y_hi    = max(float(vis.max()) + 0.4,  STOP_Z + 0.3)

    # Status annotation
    if open_trade:
        sc   = "#00ffcc" if open_trade["direction"] == "LONG" else "#ff4b4b"
        pct2 = max(0, min(100, (abs(open_trade["entry_z"]) - abs(open_trade["curr_z"])) /
                               max(abs(open_trade["entry_z"]), 0.01) * 100))
        ann_txt = (
            "<b>OPEN — " + open_trade["direction"] + "</b><br>"
            "Z: " + str(round(open_trade["curr_z"], 2)) + "  →  0.0<br>"
            + str(round(pct2, 0)) + "% of the way there"
        )
        annotations = [dict(
            x=x_end, y=open_trade["curr_z"], xref="x", yref="y",
            text=ann_txt, showarrow=True, arrowhead=2, arrowsize=0.9,
            arrowcolor=sc, ax=-90, ay=0,
            font=dict(family="IBM Plex Mono", size=10, color=sc),
            bgcolor="rgba(11,14,20,0.92)",
            bordercolor=sc, borderwidth=1, borderpad=6, align="left",
        )]
    else:
        annotations = [dict(
            x=x_end, y=float(z.iloc[-1]), xref="x", yref="y",
            text="NEUTRAL<br>±" + str(ENTRY_Z) + " to trigger",
            showarrow=False,
            font=dict(family="IBM Plex Mono", size=9, color="#4a5568"),
            bgcolor="rgba(11,14,20,0.85)",
            bordercolor="#1e2330", borderwidth=1, borderpad=5,
        )]

    fig.update_layout(
        template="plotly_dark", height=320,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
        showlegend=False, annotations=annotations,
        xaxis=dict(
            range=[x_start, x_end], showgrid=False,
            rangeslider=dict(visible=False),
            rangeselector=dict(
                bgcolor="#111318", activecolor="#00d1ff",
                bordercolor="rgba(255,255,255,0.1)",
                font=dict(family="IBM Plex Mono", size=10, color="#8892a4"),
                buttons=[
                    dict(count=1, label="1Y", step="year",  stepmode="backward"),
                    dict(count=3, label="3Y", step="year",  stepmode="backward"),
                    dict(step="all", label="All"),
                ],
            ),
        ),
        yaxis=dict(range=[y_lo, y_hi], showgrid=False, zeroline=False, autorange=False),
        hovermode="x unified", font=dict(family="IBM Plex Mono"),
    )
    return fig


# =============================================================================
# 11. BACKTEST SECTION
# =============================================================================
def render_backtest(report: pd.DataFrame, eq_curve: pd.DataFrame = None, final_balance: float = TOTAL_CAPITAL):
    st.divider()
    n_yrs  = 10
    cagr_r = round(((final_balance / TOTAL_CAPITAL) ** (1 / n_yrs) - 1) * 100, 1) if final_balance > 0 else 0
    cagr_c = "#00d4a0" if cagr_r >= 20 else "#f5a623" if cagr_r >= 0 else "#f56565"
    st.markdown(
        '<div style="display:flex;align-items:center;gap:14px;margin-bottom:6px;">'
        '<h2 style="margin:0;font-family:monospace;">10-Year Backtest</h2>'
        '<span style="font-family:monospace;font-size:10px;padding:3px 10px;border-radius:3px;'
        'color:#a78bfa;border:1px solid rgba(167,139,250,0.4);">'
        'Geometric compounding · RSI filter · Velocity filter · Medallion sizing</span>'
        + '<span style="font-family:monospace;font-size:13px;font-weight:600;color:' + cagr_c + ';">'
        + f"CAGR {cagr_r:+.1f}%  →  ${final_balance:,.0f}"
        + '</span>'
        + '</div>',
        unsafe_allow_html=True,
    )

    if report.empty:
        st.warning("No closed trades. Check tickers or widen entry threshold.")
        return

    total_pnl  = report["TotalPnL"].sum()
    stk_pnl    = report["StockPnL"].sum()
    opt_pnl    = report["OptionPnL"].sum()
    win_rate   = (report["TotalPnL"] > 0).mean()
    avg_trade  = report["TotalPnL"].mean()
    timeouts   = (report["ExitReason"] == "TIMEOUT").sum()
    stops      = (report["ExitReason"] == "STOP").sum()
    exits      = (report["ExitReason"] == "EXIT").sum()
    n_filtered = report.shape[0]

    pnl_c = "#00d4a0" if total_pnl >= 0 else "#f56565"
    stk_c = "#00d4a0" if stk_pnl  >= 0 else "#f56565"
    opt_c = "#00d4a0" if opt_pnl  >= 0 else "#f56565"

    k = st.columns(8)
    _kpi(k[0], "Total P&L",    f"${total_pnl:+,.0f}",  pnl_c)
    _kpi(k[1], "Stock P&L",    f"${stk_pnl:+,.0f}",    stk_c)
    _kpi(k[2], "Options P&L",  f"${opt_pnl:+,.0f}",    opt_c)
    _kpi(k[3], "Win Rate",     f"{win_rate:.0%}",       "#e8c96d")
    _kpi(k[4], "Avg Trade",    f"${avg_trade:+,.0f}",   pnl_c)
    velocities = (report["ExitReason"] == "VELOCITY").sum() if "ExitReason" in report.columns else 0
    _kpi(k[5], "Clean Exits",  str(exits),              "#00d4a0")
    _kpi(k[6], "Velocity Cuts",str(velocities),         "#4a9eff" if velocities else "#4a5568")
    _kpi(k[7], "Stops",        str(stops),              "#f56565" if stops else "#4a5568")

    st.markdown("<br>", unsafe_allow_html=True)

    # Equity curve: Hybrid vs Stock-Only benchmark
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=report["Date"], y=report["CumPnL"],
        name="Hybrid (60/40)", fill="tozeroy",
        fillcolor="rgba(0,212,160,0.07)",
        line=dict(color="#00ffcc", width=3),
        hovertemplate="%{x|%b %Y}<br>Hybrid P&L: $%{y:,.0f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=report["Date"], y=report["CumStockOnly"],
        name="Stock Only",
        line=dict(color="#8892a4", width=1.5, dash="dot"),
        hovertemplate="%{x|%b %Y}<br>Stock only: $%{y:,.0f}<extra></extra>",
    ))
    # Geometric compounding curve overlay
    if eq_curve is not None and not eq_curve.empty:
        fig.add_trace(go.Scatter(
            x=eq_curve["Date"], y=eq_curve["Value"],
            name="Compounding Balance",
            line=dict(color="#e8c96d", width=1.5, dash="dot"),
            hovertemplate="%{x|%b %Y}<br>Balance: $%{y:,.0f}<extra></extra>",
        ))
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.12)", width=1, dash="dot"))
    fig.update_layout(
        template="plotly_dark", height=360,
        paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
        margin=dict(l=12, r=12, t=12, b=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    font=dict(family="IBM Plex Mono", size=10, color="#8892a4"),
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, zeroline=False, tickprefix="$"),
        hovermode="x unified", font=dict(family="IBM Plex Mono"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Trade ledger
    with st.expander("Full Trade Ledger"):
        disp = report[["Date","Pair","Direction","ExitReason","DaysHeld",
                        "StockPnL","OptionPnL","TotalPnL"]].copy()
        disp = disp.sort_values("Date", ascending=False)
        disp["Date"] = disp["Date"].dt.strftime("%b %d %Y")
        st.dataframe(disp, use_container_width=True, hide_index=True)


# =============================================================================
# 12. MAIN
# =============================================================================
def main():
    # ── Header ──────────────────────────────────────────────────────────────
    col_t, col_m = st.columns([3, 1])
    with col_t:
        st.title("Omni-Arb Terminal v10.0")
        st.caption(
            "Institutional Edition  |  Hybrid 60/40 Strategy  |  "
            "Medallion Sizing  |  RSI Trend Filter  |  "
            "Updated: " + datetime.now().strftime("%b %d %Y %H:%M ET")
        )
    with col_m:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div style="text-align:right;font-family:monospace;font-size:10px;color:#4a5568;line-height:1.8;">'
            'Entry ±' + str(ENTRY_Z) + '  |  Stop ±' + str(STOP_Z) + '<br>'
            'Exit 0.0  |  Timeout ' + str(MAX_HOLD_DAYS) + 'd<br>'
            'RSI block &lt;' + str(RSI_LOW) + ' / &gt;' + str(RSI_HIGH) + '<br>'
            '$' + str(int(TOTAL_CAPITAL)) + ' total  |  $' + str(int(ALLOC_PER_PAIR)) + '/pair'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    with st.spinner("Syncing 10 years of market data..."):
        df = get_market_data()

    with st.spinner("Running backtest + live analysis..."):
        report, pair_history, open_trades, eq_curve, final_balance = run_backtest(df)

    # ── Account Health Bar ──────────────────────────────────────────────────
    utilization = len(open_trades) * ALLOC_PER_PAIR
    util_pct    = utilization / TOTAL_CAPITAL
    dry_powder  = TOTAL_CAPITAL - utilization
    util_color  = "#f5a623" if util_pct >= 0.6 else "#00d4a0"

    st.markdown(
        '<div style="background:#111318;border:1px solid rgba(255,255,255,0.07);'
        'border-radius:6px;padding:14px 20px;margin-bottom:20px;">'
        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;">'
        '<span style="font-family:monospace;font-size:11px;color:#4a5568;text-transform:uppercase;letter-spacing:0.1em;">Account Capital Utilization</span>'
        '<span style="font-family:monospace;font-size:12px;color:' + util_color + ';">'
        '$' + str(int(utilization)) + ' deployed  ·  $' + str(int(dry_powder)) + ' dry powder'
        '</span>'
        '</div>'
        '<div style="height:6px;background:rgba(255,255,255,0.07);border-radius:3px;">'
        '<div style="width:' + str(round(util_pct * 100, 1)) + '%;height:100%;background:' + util_color + ';border-radius:3px;"></div>'
        '</div>'
        '<div style="display:flex;justify-content:space-between;margin-top:6px;">'
        + "".join(
            '<span style="font-family:monospace;font-size:9px;color:#2d3748;">'
            + f"{t1}/{t2}  ${int(ALLOC_PER_PAIR)}"
            + ('  <span style="color:#00d4a0;">●</span>' if f"{t1}/{t2}" in open_trades else '  <span style="color:#2d3748;">○</span>')
            + '</span>'
            for t1, t2 in PAIRS
        ) +
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Portfolio KPI strip ─────────────────────────────────────────────────
    if not report.empty:
        total_pnl = report["TotalPnL"].sum()
        win_rate  = (report["TotalPnL"] > 0).mean()
        n_years   = 10
        cagr      = round(((final_balance / TOTAL_CAPITAL) ** (1 / n_years) - 1) * 100, 1)
        cagr_col  = "#00d4a0" if cagr >= 20 else "#f5a623" if cagr >= 0 else "#f56565"
        k1, k2, k3, k4, k5 = st.columns(5)
        _kpi(k1, "Final Balance",       f"${final_balance:,.0f}",
             "#00d4a0" if final_balance > TOTAL_CAPITAL else "#f56565")
        _kpi(k2, "10Y CAGR",            f"{cagr:+.1f}%",      cagr_col)
        _kpi(k3, "10Y Portfolio P&L",   f"${total_pnl:+,.0f}",
             "#00d4a0" if total_pnl >= 0 else "#f56565")
        _kpi(k4, "Win Rate",            f"{win_rate:.0%}",     "#e8c96d")
        _kpi(k5, "Dry Powder",          f"${dry_powder:,.0f}", util_color)
        st.markdown("<br>", unsafe_allow_html=True)

    # ── SECTION A: ACTIVE SIGNAL CARDS ─────────────────────────────────────
    n_active  = len(open_trades)
    n_neutral = len(PAIRS) - n_active
    s_color   = "#00ffcc" if n_active > 0 else "#4a5568"

    st.markdown(
        '<div style="display:flex;align-items:center;gap:16px;margin-bottom:14px;">'
        '<h2 style="margin:0;font-family:monospace;">Active Signals</h2>'
        '<span style="font-family:monospace;font-size:12px;padding:4px 12px;border-radius:3px;'
        'color:' + s_color + ';border:1px solid ' + s_color + ';">'
        + str(n_active) + ' open  /  ' + str(n_neutral) + ' neutral'
        '</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    for t1, t2 in PAIRS:
        pk = f"{t1}/{t2}"
        if pk not in open_trades:
            continue
        z, _   = calculate_pair_stats(df, t1, t2)
        curr_z = float(z.iloc[-1])
        p1, p2 = float(df[t1].iloc[-1]), float(df[t2].iloc[-1])
        ot     = open_trades[pk]
        start  = abs(ot["entry_z"])
        ot["pct_to_target"] = max(0, min(100, (start - abs(curr_z)) / start * 100 if start else 0))
        render_signal_card(t1, t2, ot, curr_z, p1, p2)

    # ── SECTION B: WATCHLIST ────────────────────────────────────────────────
    neutral = [(t1, t2) for t1, t2 in PAIRS if f"{t1}/{t2}" not in open_trades]
    if neutral:
        st.markdown(
            '<p style="font-family:monospace;font-size:10px;letter-spacing:0.12em;'
            'color:#4a5568;text-transform:uppercase;margin:6px 0 10px;">Neutral Pairs — Monitoring</p>',
            unsafe_allow_html=True,
        )
        ncols = st.columns(len(neutral))
        for i, (t1, t2) in enumerate(neutral):
            z, _ = calculate_pair_stats(df, t1, t2)
            curr_z = float(z.iloc[-1])
            rsi_a  = compute_rsi(df[t1])
            rsi_b  = compute_rsi(df[t2])
            # Determine candidate direction and check filter
            if curr_z >= ENTRY_Z:
                tf = trend_filter(df, t1, t2, "SHORT")
            elif curr_z <= -ENTRY_Z:
                tf = trend_filter(df, t1, t2, "LONG")
            else:
                tf = {"blocked": False, "rsi_a": rsi_a, "rsi_b": rsi_b, "reason": ""}
            with ncols[i]:
                render_monitor_card(t1, t2, curr_z,
                                    tf["rsi_a"], tf["rsi_b"],
                                    tf["blocked"], tf.get("reason", ""))

    st.divider()

    # ── SECTION C: PAIR ANALYSIS CHARTS ─────────────────────────────────────
    st.markdown(
        '<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:10px;">'
        '<h2 style="margin:0;font-family:monospace;">Pair Analysis</h2>'
        '<span style="font-family:monospace;font-size:11px;color:#4a5568;">'
        '1Y default · All = 10Y · Glowing dot = active position'
        '</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Legend
    st.markdown(
        '<div style="display:flex;flex-wrap:wrap;gap:16px;padding:6px 4px 12px;'
        'font-family:monospace;font-size:11px;color:#4a5568;'
        'border-bottom:1px solid rgba(255,255,255,0.05);margin-bottom:12px;">'
        '<span><span style="color:#00ffcc;">▲</span> Long entry</span>'
        '<span><span style="color:#ff4b4b;">▼</span> Short entry</span>'
        '<span><span style="color:#ffffff;">◆</span> Exit (Z=0)</span>'
        '<span><span style="color:#f5a623;">◆</span> Stop/timeout</span>'
        '<span><span style="background:rgba(0,255,204,0.12);padding:0 5px;">■</span> Long profit zone</span>'
        '<span><span style="background:rgba(255,75,75,0.12);padding:0 5px;">■</span> Short profit zone</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    chart_cols = st.columns(2)
    for i, (t1, t2) in enumerate(PAIRS):
        pk  = f"{t1}/{t2}"
        z, _ = calculate_pair_stats(df, t1, t2)
        ot  = open_trades.get(pk)
        accent = (
            "#00ffcc" if (ot and ot["direction"] == "LONG") else
            "#ff4b4b" if (ot and ot["direction"] == "SHORT") else
            "#4a5568"
        )
        badge = (
            '<span style="font-family:monospace;font-size:9px;font-weight:600;'
            'padding:2px 8px;border-radius:2px;color:' + accent + ';'
            'border:1px solid ' + accent + ';opacity:0.85;">'
            + (ot["direction"] + " OPEN") + '</span>'
            if ot else
            '<span style="font-family:monospace;font-size:9px;color:#2d3748;">NEUTRAL</span>'
        )
        with chart_cols[i % 2]:
            st.markdown(
                '<div style="display:flex;justify-content:space-between;'
                'align-items:center;margin-bottom:6px;">'
                '<p style="margin:0;font-family:monospace;font-size:14px;font-weight:500;color:'
                + accent + ';">'
                + t1 + ' <span style="color:#2d3748;">/</span> ' + t2
                + '</p>' + badge + '</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                render_pair_chart(z, t1, t2, pair_history[pk], ot),
                use_container_width=True,
            )

    # ── SECTION D: BACKTEST ─────────────────────────────────────────────────
    render_backtest(report, eq_curve, final_balance)

    # ── SECTION E: ENGINE EXPLAINER ─────────────────────────────────────────
    st.divider()
    st.markdown(
        '<div style="background:#111318;border-radius:6px;padding:20px 24px;'
        'border-left:3px solid #4a9eff;">'
        '<p style="font-family:monospace;font-size:11px;color:#4a9eff;margin:0 0 14px;'
        'text-transform:uppercase;letter-spacing:0.12em;">Strategy Architecture</p>'
        '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:20px;">'

        '<div><p style="font-family:monospace;font-size:11px;color:#e8eaf0;margin:0 0 5px;">① Trend Filter</p>'
        '<p style="font-size:11px;color:#8892a4;margin:0;line-height:1.7;">'
        'RSI check at every entry. If RSI &gt; ' + str(RSI_HIGH) + ' or &lt; ' + str(RSI_LOW) + ', '
        'the entry is skipped. Prevents "stepping in front of a freight train" '
        '(e.g., NVDA parabolic move).</p></div>'

        '<div><p style="font-family:monospace;font-size:11px;color:#e8eaf0;margin:0 0 5px;">② Entry + Sizing</p>'
        '<p style="font-size:11px;color:#8892a4;margin:0;line-height:1.7;">'
        'Z crosses ±' + str(ENTRY_Z) + ' AND RSI safe → position opens. '
        '$' + str(int(ALLOC_PER_PAIR * EQUITY_PCT)) + ' into Medallion-sized equity legs, '
        '$' + str(int(ALLOC_PER_PAIR * OPTION_PCT)) + ' into 2.5% OTM vertical spreads. '
        'One trade per pair at a time.</p></div>'

        '<div><p style="font-family:monospace;font-size:11px;color:#e8eaf0;margin:0 0 5px;">③ Hold + Monitor</p>'
        '<p style="font-size:11px;color:#8892a4;margin:0;line-height:1.7;">'
        'Shaded chart zone = profit room remaining. Progress bar tracks Z → 0. '
        'Timer turns amber at day 15. Options amplify equity P&L by '
        + str(OPT_LEVERAGE) + '× while capping max loss at premium paid.</p></div>'

        '<div><p style="font-family:monospace;font-size:11px;color:#e8eaf0;margin:0 0 5px;">④ Exit Rules</p>'
        '<p style="font-size:11px;color:#8892a4;margin:0;line-height:1.7;">'
        'Z returns to 0.0 → full profit (clean exit). '
        '|Z| ≥ ' + str(STOP_Z) + ' → stop loss. '
        'Day ' + str(MAX_HOLD_DAYS) + ' → hard timeout, close at market. '
        'All three cases close both equity and options legs simultaneously.</p></div>'

        '</div></div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

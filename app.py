# =============================================================================
# OMNI-ARB v8.0  |  Hybrid Statistical Arbitrage Terminal
# Strategy: 60/40 Equity + Vertical Options | Medallion Beta-Neutral Sizing
# Engine:   State-Machine Hysteresis | 21-Day Timeout | Rolling Beta
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
    page_title="Omni-Arb v8.0",
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
PAIRS          = [('XOM', 'CVX'), ('V', 'MA'), ('NVDA', 'AMD'), ('KO', 'PEP'), ('MSTR', 'BTC-USD')]
PORTFOLIO_TOTAL= 1000.0
EQUITY_ALLOC   = 0.60        # $600 — fractional shares, Medallion-sized
OPTION_ALLOC   = 0.40        # $400 — vertical spreads
ENTRY_Z        = 2.25
EXIT_Z         = 0.0         # strict zero-cross exit
STOP_Z         = 3.75
STRIKE_OFFSET  = 0.025       # 2.5% OTM for vertical legs
ROLL_WIN       = 60          # rolling OLS window (days)
MAX_HOLD_DAYS  = 21          # hard timeout in trading bars


# =============================================================================
# 2. DATA ENGINE
# =============================================================================
@st.cache_data(ttl=3600)
def get_market_data() -> pd.DataFrame:
    tickers = list(set(t for p in PAIRS for t in p))
    df = yf.download(tickers, period="3y", interval="1d")["Close"]
    return df.ffill().dropna()


def calculate_pair_stats(df: pd.DataFrame, t1: str, t2: str):
    """Rolling OLS beta + z-score for the log-price spread."""
    y     = np.log(df[t1])
    x     = sm.add_constant(np.log(df[t2]))
    model = RollingOLS(y, x, window=ROLL_WIN).fit()
    beta  = model.params[t2]
    const = model.params["const"]
    spread  = y - (beta * np.log(df[t2]) + const)
    z_score = (spread - spread.rolling(ROLL_WIN).mean()) / spread.rolling(ROLL_WIN).std()
    return z_score, beta


# =============================================================================
# 3. MEDALLION SIZING  (Corrects the Price-Beta Trap)
# =============================================================================
def medallion_legs(price_a: float, price_b: float, beta: float,
                   capital: float) -> dict:
    """
    Risk-neutral dollar split:
      dollar_a = capital / (1 + |β|)
      dollar_b = capital − dollar_a
    Guarantees dollar_b = β × dollar_a so both legs track
    the Z-score dollar-for-dollar.
    """
    b     = max(abs(beta), 0.01)
    d_a   = capital / (1.0 + b)
    d_b   = capital - d_a
    sa    = max(0.1, round(d_a / price_a, 1))
    sb    = max(0.1, round(d_b / price_b, 1))
    na, nb = round(sa * price_a, 2), round(sb * price_b, 2)
    return {
        "shares_a": sa, "shares_b": sb,
        "notional_a": na, "notional_b": nb,
        "total": round(na + nb, 2),
        "risk_imbal": round(na - (nb / b), 2),   # ~$0 = perfect neutral
    }


# =============================================================================
# 4. SPREAD P&L  (Log-return method — Medallion standard)
# =============================================================================
def spread_pnl(ep_a: float, ep_b: float, cp_a: float, cp_b: float,
               sa: float, beta: float, direction: str) -> float:
    """
    Measures reversion of the LOG spread, scaled by entry notional.
    Correct regardless of absolute market direction.
    """
    notional_a   = sa * ep_a
    log_s_entry  = math.log(ep_a) - beta * math.log(ep_b)
    log_s_now    = math.log(cp_a) - beta * math.log(cp_b)
    delta        = log_s_now - log_s_entry
    sign         = 1 if direction == "LONG" else -1
    return round(delta * notional_a * sign, 2)


# =============================================================================
# 5. HYBRID BACKTESTER  (state-machine, 21-day timeout, live P&L)
# =============================================================================
def run_hybrid_backtest(df: pd.DataFrame) -> tuple:
    """
    Returns:
      report      — closed-trade ledger DataFrame
      pair_trades — entry/exit dots per pair (for chart overlay)
      open_trades — currently active trades with live P&L
    """
    ledger      = []
    pair_trades = {f"{t1}/{t2}": [] for t1, t2 in PAIRS}
    open_trades = {}

    for t1, t2 in PAIRS:
        if t1 not in df.columns or t2 not in df.columns:
            continue

        z, beta_s = calculate_pair_stats(df, t1, t2)
        in_pos    = False
        entry_idx = None
        direction = None

        for i in range(ROLL_WIN, len(z)):
            curr_z    = float(z.iloc[i])
            days_held = (i - entry_idx) if in_pos else 0

            # ── ENTRY ──────────────────────────────────────────────────────
            if not in_pos:
                if curr_z >= ENTRY_Z:
                    in_pos, direction, entry_idx = True, "SHORT", i
                elif curr_z <= -ENTRY_Z:
                    in_pos, direction, entry_idx = True, "LONG",  i

            # ── EXIT CHECK ─────────────────────────────────────────────────
            else:
                hit_target  = (direction == "LONG"  and curr_z >= -EXIT_Z) or \
                              (direction == "SHORT" and curr_z <=  EXIT_Z)
                hit_stop    = abs(curr_z) >= STOP_Z
                hit_timeout = days_held >= MAX_HOLD_DAYS

                if hit_target or hit_stop or hit_timeout:
                    ep_a   = float(df[t1].iloc[entry_idx])
                    ep_b   = float(df[t2].iloc[entry_idx])
                    cp_a   = float(df[t1].iloc[i])
                    cp_b   = float(df[t2].iloc[i])
                    beta   = float(beta_s.iloc[entry_idx])
                    eq_cap = PORTFOLIO_TOTAL * EQUITY_ALLOC
                    legs   = medallion_legs(ep_a, ep_b, beta, eq_cap)

                    # Spread return (percentage)
                    ret_a   = cp_a / ep_a - 1
                    ret_b   = cp_b / ep_b - 1
                    sp_ret  = (ret_a - beta * ret_b) * (1 if direction == "LONG" else -1)

                    stk_pnl = eq_cap * sp_ret
                    opt_pnl = (PORTFOLIO_TOTAL * OPTION_ALLOC) * sp_ret * 4.5
                    exit_r  = "TIMEOUT" if hit_timeout else "STOP" if hit_stop else "EXIT"

                    pair_trades[f"{t1}/{t2}"].append({
                        "entry_date": z.index[entry_idx],
                        "exit_date":  z.index[i],
                        "entry_z":    float(z.iloc[entry_idx]),
                        "exit_z":     curr_z,
                        "dir":        direction,
                        "exit_reason":exit_r,
                    })
                    ledger.append({
                        "Date":      z.index[i],
                        "Pair":      f"{t1}/{t2}",
                        "Direction": direction,
                        "ExitReason":exit_r,
                        "DaysHeld":  days_held,
                        "StockPnL":  round(stk_pnl, 2),
                        "OptionPnL": round(opt_pnl, 2),
                        "TotalPnL":  round(stk_pnl + opt_pnl, 2),
                    })
                    in_pos = False

        # ── LIVE OPEN TRADE ─────────────────────────────────────────────────
        if in_pos:
            ep_a   = float(df[t1].iloc[entry_idx])
            ep_b   = float(df[t2].iloc[entry_idx])
            cp_a   = float(df[t1].iloc[-1])
            cp_b   = float(df[t2].iloc[-1])
            beta   = float(beta_s.iloc[entry_idx])
            eq_cap = PORTFOLIO_TOTAL * EQUITY_ALLOC
            legs   = medallion_legs(ep_a, ep_b, beta, eq_cap)
            days   = len(z) - 1 - entry_idx

            sp     = spread_pnl(ep_a, ep_b, cp_a, cp_b, legs["shares_a"], beta, direction)
            ret_a  = cp_a / ep_a - 1
            ret_b  = cp_b / ep_b - 1
            sp_ret = (ret_a - beta * ret_b) * (1 if direction == "LONG" else -1)
            opt_p  = (PORTFOLIO_TOTAL * OPTION_ALLOC) * sp_ret * 4.5

            open_trades[f"{t1}/{t2}"] = {
                "direction":   direction,
                "entry_z":     float(z.iloc[entry_idx]),
                "entry_date":  z.index[entry_idx],
                "curr_z":      float(z.iloc[-1]),
                "beta":        beta,
                "legs":        legs,
                "days_held":   days,
                "live_stk":    round(sp, 2),
                "live_opt":    round(opt_p, 2),
                "live_pnl":    round(sp + opt_p, 2),
                "entry_pa":    ep_a,
                "entry_pb":    ep_b,
            }

    # ── REPORT ──────────────────────────────────────────────────────────────
    report = pd.DataFrame(ledger).sort_values("Date") if ledger else pd.DataFrame()
    if not report.empty:
        report["CumPnL"]      = report["TotalPnL"].cumsum()
        report["CumStockOnly"]= report["StockPnL"].cumsum()

    return report, pair_trades, open_trades


# =============================================================================
# 6. HTML HELPERS
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
# 7. ACTIVE SIGNAL CARD
# =============================================================================
def render_signal_card(pair_key: str, t1: str, t2: str,
                       trade: dict, curr_z: float,
                       p1: float, p2: float):
    direction  = trade["direction"]
    color      = "#00ffcc" if direction == "LONG" else "#ff4b4b"
    pnl_color  = "#00d4a0" if trade["live_pnl"] >= 0 else "#f56565"
    bg         = "rgba(0,255,204,0.04)" if direction == "LONG" else "rgba(255,75,75,0.04)"
    border     = "rgba(0,255,204,0.25)" if direction == "LONG" else "rgba(255,75,75,0.25)"
    pnl_sign   = "+" if trade["live_pnl"] >= 0 else ""
    legs       = trade["legs"]

    eq_budget  = PORTFOLIO_TOTAL * EQUITY_ALLOC
    opt_budget = PORTFOLIO_TOTAL * OPTION_ALLOC

    # Progress bar
    start_z    = abs(trade["entry_z"])
    dist       = start_z - abs(curr_z)
    pct        = max(0.0, min(100.0, (dist / start_z * 100) if start_z else 0))
    days_left  = MAX_HOLD_DAYS - trade["days_held"]
    timeout_col= "#f5a623" if trade["days_held"] >= 15 else "#4a5568"

    # Options strikes
    if direction == "LONG":
        opt_t1 = "Call Spread  strike $" + f"{p1 * (1 + STRIKE_OFFSET):.2f}"
        opt_t2 = "Put Spread   strike $" + f"{p2 * (1 - STRIKE_OFFSET):.2f}"
        leg1_v = "BUY"
        leg2_v = "SELL"
    else:
        opt_t1 = "Put Spread   strike $" + f"{p1 * (1 - STRIKE_OFFSET):.2f}"
        opt_t2 = "Call Spread  strike $" + f"{p2 * (1 + STRIKE_OFFSET):.2f}"
        leg1_v = "SELL"
        leg2_v = "BUY"

    # Pre-compute all strings — no nested f-strings
    z_str        = str(round(curr_z, 2))
    entry_z_str  = str(round(trade["entry_z"], 2))
    pnl_str      = pnl_sign + "${:,.2f}".format(trade["live_pnl"])
    stk_str      = ("+" if trade["live_stk"] >= 0 else "") + "${:,.2f}".format(trade["live_stk"])
    opt_str      = ("+" if trade["live_opt"] >= 0 else "") + "${:,.2f}".format(trade["live_opt"])
    days_str     = str(trade["days_held"]) + " / " + str(MAX_HOLD_DAYS) + "d"
    pct_str      = str(round(pct, 0)) + "% to 0.0"
    sa_str       = str(legs["shares_a"])
    sb_str       = str(legs["shares_b"])
    na_str       = "$" + "{:,.0f}".format(legs["notional_a"])
    nb_str       = "$" + "{:,.0f}".format(legs["notional_b"])
    ri_str       = ("+" if legs["risk_imbal"] >= 0 else "") + "${:,.2f}".format(legs["risk_imbal"])
    ri_col       = "#00d4a0" if abs(legs["risk_imbal"]) < 20 else "#f5a623"
    ri_label     = "✓ neutral" if abs(legs["risk_imbal"]) < 20 else "⚠ check"
    pa_str       = "$" + str(round(trade["entry_pa"], 2))
    pb_str       = "$" + str(round(trade["entry_pb"], 2))
    curr_pa_str  = "$" + str(round(p1, 2))
    curr_pb_str  = "$" + str(round(p2, 2))

    html = (
        # ── Card wrapper
        '<div style="background:' + bg + ';border:1px solid ' + border + ';'
        'border-top:3px solid ' + color + ';border-radius:6px;padding:16px;margin-bottom:10px;">'

        # ── Header
        '<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:10px;">'
        '<div>'
        '<p style="margin:0 0 2px;font-family:monospace;font-size:16px;font-weight:600;color:' + color + ';">'
        + t1 + ' <span style="color:#2d3748;">/</span> ' + t2 + '</p>'
        '<span style="font-family:monospace;font-size:10px;color:#8892a4;">'
        + direction + ' SPREAD — ' + ('Buying A / Selling B' if direction == 'LONG' else 'Selling A / Buying B')
        + '</span>'
        '</div>'
        '<div style="text-align:right;">'
        '<p style="margin:0 0 1px;font-size:9px;color:#4a5568;font-family:monospace;letter-spacing:0.1em;">LIVE P&L</p>'
        '<p style="margin:0;font-family:monospace;font-size:22px;font-weight:600;color:' + pnl_color + ';">'
        + pnl_str + '</p>'
        '<p style="margin:1px 0 0;font-size:10px;color:#4a5568;font-family:monospace;">'
        'Stk: ' + stk_str + '  Opt: ' + opt_str + '</p>'
        '</div>'
        '</div>'

        # ── Progress + timer row
        '<div style="background:rgba(0,0,0,0.2);border-radius:4px;padding:8px 12px;margin-bottom:10px;">'
        '<div style="display:flex;justify-content:space-between;margin-bottom:3px;">'
        '<span style="font-size:9px;color:#4a5568;font-family:monospace;">Entry Z: ' + entry_z_str + '</span>'
        '<span style="font-size:9px;color:' + color + ';font-family:monospace;font-weight:600;">' + pct_str + '</span>'
        '<span style="font-size:9px;color:' + timeout_col + ';font-family:monospace;">⏱ ' + days_str + '</span>'
        '</div>'
        '<div style="height:4px;background:rgba(255,255,255,0.07);border-radius:2px;">'
        '<div style="width:' + str(round(pct, 1)) + '%;height:100%;background:' + color + ';border-radius:2px;"></div>'
        '</div>'
        '<div style="display:flex;justify-content:space-between;margin-top:3px;">'
        '<span style="font-size:9px;color:#4a5568;font-family:monospace;">Z now: ' + z_str + '</span>'
        '<span style="font-size:9px;color:#4a5568;font-family:monospace;">' + str(days_left) + 'd left before timeout</span>'
        '</div>'
        '</div>'

        # ── Two-column execution grid
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;">'

        # Left: equities
        '<div style="background:rgba(0,0,0,0.25);border-radius:4px;padding:10px 12px;'
        'border:1px solid rgba(255,255,255,0.05);">'
        '<p style="margin:0 0 6px;font-size:9px;color:#4a9eff;font-family:monospace;'
        'text-transform:uppercase;letter-spacing:0.1em;">① Equities — ${:.0f}'.format(eq_budget) + '</p>'
        + _row(leg1_v + " " + t1, sa_str + " shs  " + pa_str + " → " + curr_pa_str + "  = " + na_str,
               "#00ffcc" if direction == "LONG" else "#ff4b4b")
        + _row(leg2_v + " " + t2, sb_str + " shs  " + pb_str + " → " + curr_pb_str + "  = " + nb_str,
               "#ff4b4b" if direction == "LONG" else "#00ffcc")
        + _row("Total Deployed", "$" + "{:,.0f}".format(legs["total"]))
        + _row("Risk Imbalance", ri_str + "  " + ri_label, ri_col)
        + '</div>'

        # Right: options
        '<div style="background:rgba(0,0,0,0.25);border-radius:4px;padding:10px 12px;'
        'border:1px solid rgba(255,255,255,0.05);">'
        '<p style="margin:0 0 6px;font-size:9px;color:#a78bfa;font-family:monospace;'
        'text-transform:uppercase;letter-spacing:0.1em;">② Verticals — ${:.0f}'.format(opt_budget) + '</p>'
        + _row(t1, opt_t1, "#a78bfa")
        + _row(t2, opt_t2, "#a78bfa")
        + '<div style="margin-top:8px;padding-top:6px;border-top:1px solid rgba(255,255,255,0.05);">'
        '<p style="margin:0 0 3px;font-size:10px;color:#8892a4;line-height:1.5;">'
        'Max risk = premium paid. Profit capped at spread width.<br>'
        '<span style="color:#f5a623;">Stop: |Z| ≥ ' + str(STOP_Z) + '  —  Timeout: day ' + str(MAX_HOLD_DAYS) + '</span>'
        '</p>'
        '</div>'
        '</div>'

        '</div></div>'
    )
    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# 8. MONITORING CARD  (neutral pairs)
# =============================================================================
def render_monitor_card(t1: str, t2: str, curr_z: float):
    z_col  = "#4a9eff" if curr_z > 0 else "#00ffcc"
    dist   = min(abs(ENTRY_Z - curr_z), abs(-ENTRY_Z - curr_z))
    pct    = min(abs(curr_z) / ENTRY_Z * 100, 100)
    wait   = ("Wait for +" + str(ENTRY_Z) + " to Short") if curr_z > 0 else \
             ("Dormant — near zero" if abs(curr_z) < 0.3 else "Wait for -" + str(ENTRY_Z) + " to Buy")
    st.markdown(
        '<div style="background:#111318;border:1px solid rgba(255,255,255,0.07);'
        'border-radius:4px;padding:10px 12px;height:100%;">'
        '<p style="margin:0 0 2px;font-family:monospace;font-size:13px;font-weight:500;color:#e8eaf0;">'
        + t1 + ' <span style="color:#2d3748;">/</span> ' + t2 + '</p>'
        '<p style="margin:0 0 6px;font-size:10px;color:#4a5568;font-family:monospace;">' + wait + '</p>'
        '<p style="margin:0 0 5px;font-family:monospace;font-size:22px;font-weight:600;color:' + z_col + ';">'
        + str(round(curr_z, 2)) + '</p>'
        '<div style="height:3px;background:rgba(255,255,255,0.07);border-radius:2px;margin-bottom:5px;">'
        '<div style="width:' + str(round(pct, 1)) + '%;height:100%;background:' + z_col + ';border-radius:2px;"></div>'
        '</div>'
        '<span style="font-size:9px;color:#4a5568;font-family:monospace;">'
        + str(round(dist, 2)) + 'σ from entry threshold</span>'
        '</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# 9. PAIR CHART  (z-score + entry/exit dots + active zone shading)
# =============================================================================
def render_pair_chart(df: pd.DataFrame, t1: str, t2: str,
                      z: pd.Series, history: list, open_trade: dict | None):
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
    for t in history:
        ec = "#00ffcc" if t["dir"] == "LONG" else "#ff4b4b"
        xs = "x" if t["exit_reason"] == "STOP" else "diamond-open" if t["exit_reason"] == "TIMEOUT" else "x"
        fig.add_trace(go.Scatter(
            x=[t["entry_date"]], y=[t["entry_z"]], mode="markers",
            marker=dict(color=ec, size=9, symbol="triangle-up" if t["dir"] == "LONG" else "triangle-down",
                        line=dict(color="#0b0e14", width=1.5)),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=[t["exit_date"]], y=[t["exit_z"]], mode="markers",
            marker=dict(color="#ffffff" if t["exit_reason"] == "EXIT" else
                                "#f5a623" if t["exit_reason"] == "STOP" else "#f56565",
                        size=7, symbol="diamond"),
            showlegend=False, hoverinfo="skip",
        ))

    # Glowing active entry marker
    if open_trade:
        glow = "#00ffcc" if open_trade["direction"] == "LONG" else "#ff4b4b"
        glow_o = "rgba(0,255,204,0.15)" if open_trade["direction"] == "LONG" else "rgba(255,75,75,0.15)"
        glow_m = "rgba(0,255,204,0.30)" if open_trade["direction"] == "LONG" else "rgba(255,75,75,0.30)"
        glow_s = "triangle-up" if open_trade["direction"] == "LONG" else "triangle-down"
        for sz, col in [(32, glow_o), (21, glow_m)]:
            fig.add_trace(go.Scatter(
                x=[open_trade["entry_date"]], y=[open_trade["entry_z"]], mode="markers",
                marker=dict(color=col, size=sz, symbol="circle"),
                showlegend=False, hoverinfo="skip",
            ))
        fig.add_trace(go.Scatter(
            x=[open_trade["entry_date"]], y=[open_trade["entry_z"]], mode="markers",
            marker=dict(color=glow, size=13, symbol=glow_s, line=dict(color="#0b0e14", width=2)),
            name="Active Entry",
            hovertemplate=(
                "<b>ACTIVE " + open_trade["direction"] + "</b><br>"
                + open_trade["entry_date"].strftime("%b %d, %Y")
                + "  Z at entry: " + str(round(open_trade["entry_z"], 2))
                + "<br>Z now: " + str(round(open_trade["curr_z"], 2))
                + "<extra></extra>"
            ),
        ))

    # Threshold lines
    fig.add_hline(y= ENTRY_Z,  line=dict(color="#ff4b4b", width=1.5),
                  annotation_text="SHORT (+2.25)", annotation_font=dict(color="#ff4b4b", size=9),
                  annotation_position="top left")
    fig.add_hline(y=-ENTRY_Z,  line=dict(color="#00ffcc", width=1.5),
                  annotation_text="LONG (−2.25)",  annotation_font=dict(color="#00ffcc", size=9),
                  annotation_position="bottom left")
    fig.add_hline(y=EXIT_Z,    line=dict(color="#ffffff", width=1.2, dash="dot"),
                  annotation_text="TARGET (0.0)", annotation_font=dict(color="#8892a4", size=9),
                  annotation_position="bottom right")
    fig.add_hline(y= STOP_Z,   line=dict(color="#f5a623", width=1.8, dash="dash"),
                  annotation_text="STOP (+3.75)", annotation_font=dict(color="#f5a623", size=9),
                  annotation_position="top right")
    fig.add_hline(y=-STOP_Z,   line=dict(color="#f5a623", width=1.8, dash="dash"),
                  annotation_text="STOP (−3.75)", annotation_font=dict(color="#f5a623", size=9),
                  annotation_position="bottom right")

    # X-axis: 1-year default, 15% right pad
    last     = z.index[-1]
    x_start  = max(last - pd.DateOffset(years=1), z.index[0])
    x_end    = last + (last - x_start) * 0.15
    vis      = z[z.index >= x_start]
    y_lo     = min(float(vis.min()) - 0.4, -STOP_Z - 0.3)
    y_hi     = max(float(vis.max()) + 0.4,  STOP_Z + 0.3)

    # Status annotation
    if open_trade:
        sc   = "#00ffcc" if open_trade["direction"] == "LONG" else "#ff4b4b"
        pct2 = open_trade.get("pct_to_target", 0)
        ann  = (
            "<b>OPEN — " + open_trade["direction"] + "</b><br>"
            "Z: " + str(round(open_trade["curr_z"], 2)) + "  →  0.0<br>"
            + str(round(pct2, 0)) + "% of the way there"
        )
        annotations = [dict(
            x=x_end, y=open_trade["curr_z"], xref="x", yref="y",
            text=ann, showarrow=True, arrowhead=2, arrowsize=0.9,
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
        template="plotly_dark", height=340,
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
                    dict(count=6,  label="6M", step="month", stepmode="backward"),
                    dict(count=1,  label="1Y", step="year",  stepmode="backward"),
                    dict(step="all", label="All"),
                ],
            ),
        ),
        yaxis=dict(range=[y_lo, y_hi], showgrid=False, zeroline=False, autorange=False),
        hovermode="x unified", font=dict(family="IBM Plex Mono"),
    )
    return fig


# =============================================================================
# 10. BACKTEST SECTION
# =============================================================================
def render_backtest(report: pd.DataFrame):
    st.divider()
    st.markdown(
        '<div style="display:flex;align-items:center;gap:14px;margin-bottom:4px;">'
        '<h2 style="margin:0;font-family:monospace;">Hybrid Strategy Backtest</h2>'
        '<span style="font-family:monospace;font-size:10px;padding:3px 10px;border-radius:3px;'
        'color:#a78bfa;border:1px solid rgba(167,139,250,0.4);">'
        '3yr · 60/40 · 21d timeout · Medallion sizing</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    if report.empty:
        st.warning("No closed trades in backtest window. Check tickers and data range.")
        return

    total_pnl = report["TotalPnL"].sum()
    stk_pnl   = report["StockPnL"].sum()
    opt_pnl   = report["OptionPnL"].sum()
    win_rate  = (report["TotalPnL"] > 0).mean()
    avg_trade = report["TotalPnL"].mean()
    timeouts  = (report["ExitReason"] == "TIMEOUT").sum()
    stops     = (report["ExitReason"] == "STOP").sum()
    exits     = (report["ExitReason"] == "EXIT").sum()

    pnl_c  = "#00d4a0" if total_pnl >= 0 else "#f56565"
    stk_c  = "#00d4a0" if stk_pnl  >= 0 else "#f56565"
    opt_c  = "#00d4a0" if opt_pnl  >= 0 else "#f56565"

    k = st.columns(8)
    _kpi(k[0], "Total P&L",    f"${total_pnl:+,.0f}",   pnl_c)
    _kpi(k[1], "Stock P&L",    f"${stk_pnl:+,.0f}",     stk_c)
    _kpi(k[2], "Options P&L",  f"${opt_pnl:+,.0f}",     opt_c)
    _kpi(k[3], "Win Rate",     f"{win_rate:.0%}",        "#e8c96d")
    _kpi(k[4], "Avg Trade",    f"${avg_trade:+,.0f}",    pnl_c)
    _kpi(k[5], "Clean Exits",  str(exits),               "#00d4a0")
    _kpi(k[6], "Timeouts",     str(timeouts),            "#f5a623" if timeouts else "#4a5568")
    _kpi(k[7], "Stops",        str(stops),               "#f56565" if stops else "#4a5568")

    st.markdown("<br>", unsafe_allow_html=True)

    # Equity curve
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=report["Date"], y=report["CumPnL"],
        name="Hybrid (60/40)", fill="tozeroy",
        fillcolor="rgba(0,212,160,0.07)",
        line=dict(color="#00ffcc", width=3),
    ))
    fig.add_trace(go.Scatter(
        x=report["Date"], y=report["CumStockOnly"],
        name="Stock Only",
        line=dict(color="#8892a4", width=1.5, dash="dot"),
    ))
    fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"))
    fig.update_layout(
        template="plotly_dark", height=380,
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
        display = report[["Date", "Pair", "Direction", "ExitReason",
                           "DaysHeld", "StockPnL", "OptionPnL", "TotalPnL"]].copy()
        display = display.sort_values("Date", ascending=False)
        display["Date"] = display["Date"].dt.strftime("%b %d %Y")
        st.dataframe(display, use_container_width=True, hide_index=True)


# =============================================================================
# 11. MAIN
# =============================================================================
def main():
    # Header
    col_t, col_m = st.columns([3, 1])
    with col_t:
        st.title("Omni-Arb Terminal v8.0")
        st.caption(
            "Strategy: Hybrid 60/40 Equities + Vertical Options  |  "
            "Sizing: Medallion Beta-Neutral  |  "
            "Updated: " + datetime.now().strftime("%b %d %Y %H:%M ET")
        )
    with col_m:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div style="text-align:right;font-family:monospace;font-size:10px;color:#4a5568;line-height:1.8;">'
            'Entry ±' + str(ENTRY_Z) + '  |  Stop ±' + str(STOP_Z) + '<br>'
            'Exit 0.0  |  Timeout ' + str(MAX_HOLD_DAYS) + 'd<br>'
            'Equity ' + str(int(EQUITY_ALLOC * 100)) + '% / Options ' + str(int(OPTION_ALLOC * 100)) + '%'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    with st.spinner("Syncing market data..."):
        df = get_market_data()

    with st.spinner("Running hybrid backtest..."):
        report, pair_trades, open_trades = run_hybrid_backtest(df)

    # ── SECTION A: ACTIVE SIGNALS ────────────────────────────────────────
    n_active  = len(open_trades)
    n_neutral = len(PAIRS) - n_active
    s_color   = "#00ffcc" if n_active > 0 else "#4a5568"

    st.markdown(
        '<div style="display:flex;align-items:center;gap:16px;margin-bottom:16px;">'
        '<h2 style="margin:0;font-family:monospace;">Active Signals</h2>'
        '<span style="font-family:monospace;font-size:12px;padding:4px 12px;border-radius:3px;'
        'color:' + s_color + ';border:1px solid ' + s_color + ';">'
        + str(n_active) + ' open  /  ' + str(n_neutral) + ' neutral'
        '</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Active signal cards (full width, stacked)
    for t1, t2 in PAIRS:
        pk = f"{t1}/{t2}"
        if pk not in open_trades:
            continue
        z, _ = calculate_pair_stats(df, t1, t2)
        curr_z = float(z.iloc[-1])
        p1, p2 = float(df[t1].iloc[-1]), float(df[t2].iloc[-1])
        # Attach progress pct to open_trade for chart annotation
        ot = open_trades[pk]
        start = abs(ot["entry_z"])
        ot["pct_to_target"] = max(0, min(100, (start - abs(curr_z)) / start * 100 if start else 0))
        render_signal_card(pk, t1, t2, ot, curr_z, p1, p2)

    # ── SECTION B: WATCHLIST (neutral pairs) ─────────────────────────────
    neutral = [(t1, t2) for t1, t2 in PAIRS if f"{t1}/{t2}" not in open_trades]
    if neutral:
        st.markdown(
            '<p style="font-family:monospace;font-size:10px;letter-spacing:0.12em;'
            'color:#4a5568;text-transform:uppercase;margin:6px 0 10px;">Neutral — Monitoring</p>',
            unsafe_allow_html=True,
        )
        ncols = st.columns(len(neutral))
        for i, (t1, t2) in enumerate(neutral):
            z, _ = calculate_pair_stats(df, t1, t2)
            with ncols[i]:
                render_monitor_card(t1, t2, float(z.iloc[-1]))

    st.divider()

    # ── SECTION C: PAIR ANALYSIS CHARTS ──────────────────────────────────
    st.markdown(
        '<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:10px;">'
        '<h2 style="margin:0;font-family:monospace;">Pair Analysis</h2>'
        '<span style="font-family:monospace;font-size:11px;color:#4a5568;">'
        '1-year default — use range buttons to zoom'
        '</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Chart legend
    st.markdown(
        '<div style="display:flex;flex-wrap:wrap;gap:16px;padding:6px 4px 12px;'
        'font-family:monospace;font-size:11px;color:#4a5568;'
        'border-bottom:1px solid rgba(255,255,255,0.05);margin-bottom:12px;">'
        '<span><span style="color:#00ffcc;">▲</span> Long entry</span>'
        '<span><span style="color:#ff4b4b;">▼</span> Short entry</span>'
        '<span><span style="color:#ffffff;">◆</span> Exit (target)</span>'
        '<span><span style="color:#f5a623;">◆</span> Stop/timeout</span>'
        '<span><span style="background:rgba(0,255,204,0.12);padding:0 5px;">■</span> Teal zone = long profit room</span>'
        '<span><span style="background:rgba(255,75,75,0.12);padding:0 5px;">■</span> Red zone = short profit room</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    chart_cols = st.columns(2)
    for i, (t1, t2) in enumerate(PAIRS):
        pk = f"{t1}/{t2}"
        z, _ = calculate_pair_stats(df, t1, t2)
        ot   = open_trades.get(pk)
        accent = (
            "#00ffcc" if (ot and ot["direction"] == "LONG") else
            "#ff4b4b" if (ot and ot["direction"] == "SHORT") else
            "#4a5568"
        )
        badge = (
            '<span style="font-family:monospace;font-size:9px;font-weight:600;'
            'letter-spacing:0.12em;padding:2px 8px;border-radius:2px;'
            'color:' + accent + ';border:1px solid ' + accent + ';opacity:0.85;">'
            + (ot["direction"] + " OPEN") + '</span>'
            if ot else
            '<span style="font-family:monospace;font-size:9px;color:#2d3748;">NEUTRAL</span>'
        )
        with chart_cols[i % 2]:
            st.markdown(
                '<div style="display:flex;justify-content:space-between;'
                'align-items:center;margin-bottom:6px;">'
                '<p style="margin:0;font-family:monospace;font-size:14px;font-weight:500;'
                'color:' + accent + ';">'
                + t1 + ' <span style="color:#2d3748;">/</span> ' + t2
                + '</p>' + badge + '</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                render_pair_chart(df, t1, t2, z, pair_trades[pk], ot),
                use_container_width=True,
            )

    # ── SECTION D: BACKTEST RESULTS ───────────────────────────────────────
    render_backtest(report)

    # ── SECTION E: ENGINE EXPLAINER ──────────────────────────────────────
    st.divider()
    st.markdown(
        '<div style="background:#111318;border-radius:6px;padding:20px 24px;'
        'border-left:3px solid #4a9eff;">'
        '<p style="font-family:monospace;font-size:11px;color:#4a9eff;margin:0 0 14px;'
        'text-transform:uppercase;letter-spacing:0.12em;">How the Hybrid Engine Works</p>'
        '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:20px;">'

        '<div><p style="font-family:monospace;font-size:11px;color:#e8eaf0;margin:0 0 5px;">① Entry</p>'
        '<p style="font-size:11px;color:#8892a4;margin:0;line-height:1.7;">'
        'Z crosses ±' + str(ENTRY_Z) + '. One position opens. All re-crosses ignored '
        '(toggle logic). 60% into fractional equity legs, 40% into vertical spreads.</p></div>'

        '<div><p style="font-family:monospace;font-size:11px;color:#e8eaf0;margin:0 0 5px;">② Hold</p>'
        '<p style="font-size:11px;color:#8892a4;margin:0;line-height:1.7;">'
        'Shaded zone shows remaining profit room. Progress bar tracks Z → 0. '
        'Equity side captures spread P&L; options side amplifies it 4.5× '
        'while capping max loss at premium paid.</p></div>'

        '<div><p style="font-family:monospace;font-size:11px;color:#e8eaf0;margin:0 0 5px;">③ Exit</p>'
        '<p style="font-size:11px;color:#8892a4;margin:0;line-height:1.7;">'
        'Z crosses 0.0 → full profit. |Z| ≥ ' + str(STOP_Z) + ' → stop loss. '
        'Day ' + str(MAX_HOLD_DAYS) + ' → hard timeout regardless of Z. '
        'All three cases close both legs simultaneously.</p></div>'

        '<div><p style="font-family:monospace;font-size:11px;color:#e8eaf0;margin:0 0 5px;">④ Sizing</p>'
        '<p style="font-size:11px;color:#8892a4;margin:0;line-height:1.7;">'
        'Medallion formula: dollar_a = capital/(1+β), dollar_b = capital−dollar_a. '
        'Corrects the Price-Beta Trap so both legs track Z dollar-for-dollar. '
        '0.1-share precision.</p></div>'

        '</div></div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

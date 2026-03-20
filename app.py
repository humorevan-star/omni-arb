# =============================================================================
# OMNI-ARB v5.7  |  Statistical Arbitrage Terminal
# State-machine logic: Entry Hysteresis / Toggle Logic
# Focus: Execution-Centric Signals (Stocks & Options)
# Enhanced: Rich UX, clear explanations, active-trade zone charts
# =============================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
from datetime import datetime

# =============================================================================
# 0. CONFIG & THEME
# =============================================================================
st.set_page_config(
    page_title="Omni-Arb v5.7",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .main { background-color: #0b0e14; color: #e1e1e1; }
    .stPlotlyChart { background-color: #0b0e14; border-radius: 8px; }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.5px; }
    code { color: #00d1ff !important; background: rgba(0,209,255,0.08) !important;
           padding: 2px 6px !important; border-radius: 3px !important; font-size: 12px !important; }
    .stSpinner > div { border-top-color: #00ffcc !important; }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# 1. PARAMETERS
# =============================================================================
PAIRS            = [('XOM', 'CVX'), ('V', 'MA'), ('NVDA', 'AMD'), ('KO', 'PEP'), ('GOOGL', 'META')]
ENTRY_Z          = 2.25
STOP_Z           = 3.5
EXIT_Z           = 0.0
ROLLING_WINDOW   = 60
STARTING_CAPITAL = 1000


# =============================================================================
# 2. SIZING
# =============================================================================
def compute_legs(sig, capital=STARTING_CAPITAL):
    """
    Medallion Delta-Neutral Sizing — fixes the "Price-Beta Trap".

    The naive formula (qty_b = qty_a * beta) confuses share counts with
    dollar exposure. If GOOGL=$175 and META=$580, the same number of shares
    represents very different dollar risk.

    Correct formula:
      dollar_a = capital / (1 + beta)          <- beta-weighted dollar split
      dollar_b = capital - dollar_a
      shares_a = dollar_a / price_a
      shares_b = dollar_b / price_b

    This guarantees:  dollar_b = beta * dollar_a
    So a 1-unit Z-score move produces the same P&L on both legs.

    Risk imbalance check:
      risk_imbalance = val_a - (val_b / beta)  <- should be near $0
    """
    price_a = float(sig["price_a"])
    price_b = float(sig["price_b"])
    beta    = max(abs(float(sig["beta"])), 0.01)   # floor at 0.01 — avoid /0

    # Step 1 — risk-neutral dollar allocation
    dollar_a = capital / (1.0 + beta)
    dollar_b = capital - dollar_a

    # Step 2 — convert dollars to shares at 0.1 precision
    shares_a = max(0.1, round(dollar_a / price_a, 1))
    shares_b = max(0.1, round(dollar_b / price_b, 1))

    # Step 3 — actual notionals after rounding
    notional_a = round(shares_a * price_a, 2)
    notional_b = round(shares_b * price_b, 2)
    total_cost = round(notional_a + notional_b, 2)

    # Risk imbalance: val_a - (val_b / beta) should be ~$0 if perfectly neutral
    risk_imbalance = round(notional_a - (notional_b / beta), 2)
    # Dollar imbalance (absolute, for colour threshold in card)
    dollar_imbal   = round(abs(notional_a - notional_b), 2)
    # Signed dollar imbalance for display
    imbalance      = round(notional_a - notional_b, 2)
    hedge_ratio    = round(shares_a / shares_b, 2) if shares_b != 0 else 0
    ratio_str      = str(shares_a) + ":" + str(shares_b)

    # Step 4 — unrealised P&L from entry prices (state-machine replay)
    pnl = pnl_a = pnl_b = None
    entry_pa = entry_pb = None
    ot = sig.get("open_trade")
    if ot and ot.get("entry_price_a") and ot.get("entry_price_b"):
        ep_a     = float(ot["entry_price_a"])
        ep_b     = float(ot["entry_price_b"])
        entry_pa = round(ep_a, 2)
        entry_pb = round(ep_b, 2)
        is_long  = ot["direction"] == "LONG"
        if is_long:
            pnl_a = round((price_a - ep_a) * shares_a, 2)
            pnl_b = round(-(price_b - ep_b) * shares_b, 2)
        else:
            pnl_a = round(-(price_a - ep_a) * shares_a, 2)
            pnl_b = round((price_b - ep_b) * shares_b, 2)
        pnl = round(pnl_a + pnl_b, 2)

    return {
        # Core sizing
        "shares_a":       shares_a,
        "shares_b":       shares_b,
        "ratio":          ratio_str,
        # Notionals
        "notional_a":     notional_a,
        "notional_b":     notional_b,
        "total_cost":     total_cost,      # backward compat alias
        "total_deployed": total_cost,
        # Neutrality metrics
        "risk_imbalance": risk_imbalance,  # ~$0 = perfect delta-neutral
        "imbalance":      imbalance,       # signed dollar imbalance
        "dollar_imbal":   dollar_imbal,    # abs, for card colour threshold
        "hedge_ratio":    hedge_ratio,
        # P&L
        "pnl":            pnl,
        "pnl_a":          pnl_a,
        "pnl_b":          pnl_b,
        # Entry prices for P&L calculation
        "entry_pa":       entry_pa,
        "entry_pb":       entry_pb,
    }


# =============================================================================
# 3. DATA ENGINE  (State-Machine Processor)
# =============================================================================
@st.cache_data(ttl=3600)
def get_market_data():
    tickers = list(set([t for p in PAIRS for t in p]))
    data = yf.download(tickers, period="750d", interval="1d")["Close"]
    return data


def process_pairs(df_raw):
    processed, active = [], []
    for ticker_a, ticker_b in PAIRS:
        if ticker_a not in df_raw.columns or ticker_b not in df_raw.columns:
            continue
        pair_df = df_raw[[ticker_a, ticker_b]].dropna()
        y = np.log(pair_df[ticker_a])
        x = sm.add_constant(np.log(pair_df[ticker_b]))

        model  = RollingOLS(y, x, window=ROLLING_WINDOW).fit()
        betas  = model.params[ticker_b]
        consts = model.params["const"]
        spread = y - (betas * np.log(pair_df[ticker_b]) + consts)
        spread = spread.dropna()

        z_series = (
            (spread - spread.rolling(ROLLING_WINDOW).mean())
            / spread.rolling(ROLLING_WINDOW).std()
        ).dropna()

        curr_z   = z_series.iloc[-1]
        adf_pval = adfuller(spread)[1]

        in_open          = False
        open_direction   = None
        open_entry_z     = None
        open_entry_date  = None
        open_entry_pa    = None   # price of leg A at entry
        open_entry_pb    = None   # price of leg B at entry

        for _i in range(1, len(z_series)):
            _zp = z_series.iloc[_i - 1]
            _zc = z_series.iloc[_i]
            _dt = z_series.index[_i]

            if not in_open:
                if _zp > -ENTRY_Z and _zc <= -ENTRY_Z:
                    in_open          = True
                    open_direction   = "LONG"
                    open_entry_z     = _zc
                    open_entry_date  = _dt
                    open_entry_pa    = pair_df[ticker_a].loc[_dt] if _dt in pair_df.index else None
                    open_entry_pb    = pair_df[ticker_b].loc[_dt] if _dt in pair_df.index else None
                elif _zp < ENTRY_Z and _zc >= ENTRY_Z:
                    in_open          = True
                    open_direction   = "SHORT"
                    open_entry_z     = _zc
                    open_entry_date  = _dt
                    open_entry_pa    = pair_df[ticker_a].loc[_dt] if _dt in pair_df.index else None
                    open_entry_pb    = pair_df[ticker_b].loc[_dt] if _dt in pair_df.index else None
            else:
                closed = (
                    (open_direction == "LONG"  and _zp < 0 and _zc >= 0) or
                    (open_direction == "SHORT" and _zp > 0 and _zc <= 0) or
                    abs(_zc) >= STOP_Z
                )
                if closed:
                    in_open          = False
                    open_direction   = None
                    open_entry_z     = None
                    open_entry_date  = None
                    open_entry_pa    = None
                    open_entry_pb    = None

        pct_to_target = 0.0
        if in_open and open_entry_z and open_entry_z != 0:
            pct_to_target = round(
                (abs(open_entry_z) - abs(curr_z)) / abs(open_entry_z) * 100, 1
            )

        open_trade = (
            {
                "direction":    open_direction,
                "entry_z":      open_entry_z,
                "entry_date":   open_entry_date,
                "curr_z":       curr_z,
                "pct_to_target": pct_to_target,
                "entry_price_a": open_entry_pa,
                "entry_price_b": open_entry_pb,
            }
            if in_open else None
        )

        direction_now = open_direction if open_trade else "NEUTRAL"

        info = {
            "pair":            str(ticker_a) + "/" + str(ticker_b),
            "a":               ticker_a,
            "b":               ticker_b,
            "price_a":         pair_df[ticker_a].iloc[-1],
            "price_b":         pair_df[ticker_b].iloc[-1],
            "curr_z":          curr_z,
            "beta":            betas.iloc[-1],
            "z_series":        z_series,
            "is_cointegrated": adf_pval < 0.05,
            "adf_pval":        adf_pval,
            "open_trade":      open_trade,
            "direction":       direction_now,
        }
        processed.append(info)
        if info["direction"] != "NEUTRAL":
            active.append(info)

    return processed, active


# =============================================================================
# 4. HTML HELPERS
# =============================================================================
def _row(label, value, val_color="#e8eaf0"):
    return (
        '<div style="display:flex;justify-content:space-between;align-items:center;'
        'padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.04);">'
        '<span style="font-size:11px;color:#4a5568;font-family:monospace;">' + label + '</span>'
        '<span style="font-family:monospace;font-size:12px;font-weight:500;color:' + val_color + ';">' + value + '</span>'
        '</div>'
    )


# =============================================================================
# 5. ACTIVE SIGNAL CARD
# =============================================================================
def render_trade_card(sig):
    is_long     = sig["direction"] == "LONG"
    accent      = "#00ffcc" if is_long else "#ff4b4b"
    bg          = "rgba(0,255,204,0.04)"  if is_long else "rgba(255,75,75,0.04)"
    border      = "rgba(0,255,204,0.25)"  if is_long else "rgba(255,75,75,0.25)"
    leg1_action = "BUY"  if is_long else "SELL"
    leg2_action = "SELL" if is_long else "BUY"
    leg1_col    = "#00ffcc" if is_long else "#ff4b4b"
    leg2_col    = "#ff4b4b" if is_long else "#00ffcc"
    legs        = compute_legs(sig)
    ot          = sig.get("open_trade") or {}

    z_str       = str(round(sig["curr_z"], 2))
    beta_str    = str(round(sig["beta"], 2))
    entry_z_str = str(round(ot.get("entry_z", 0), 2)) if ot else "N/A"
    pct_raw     = ot.get("pct_to_target", 0) if ot else 0
    pct_clamped = max(0, min(float(pct_raw), 100))
    pct_str     = str(round(pct_clamped, 0)) + "%"
    pa_str      = "$" + str(round(sig["price_a"], 2))
    pb_str      = "$" + str(round(sig["price_b"], 2))
    sha_str     = str(legs["shares_a"])   # already float with 1dp
    shb_str     = str(legs["shares_b"])
    not_a_str   = "$" + "{:,.0f}".format(legs["notional_a"])
    not_b_str   = "$" + "{:,.0f}".format(legs["notional_b"])
    total_str   = "$" + "{:,.0f}".format(legs["total_cost"])
    hedge_str   = str(legs["shares_a"]) + ":" + str(legs["shares_b"])

    # Entry prices & P&L strings
    epa_str     = "$" + str(round(legs["entry_pa"], 2)) if legs["entry_pa"] else None
    epb_str     = "$" + str(round(legs["entry_pb"], 2)) if legs["entry_pb"] else None
    pnl_val     = legs["pnl"]
    pnl_a_val   = legs["pnl_a"]
    pnl_b_val   = legs["pnl_b"]
    has_pnl     = pnl_val is not None

    def fmt_pnl(v):
        if v is None: return ""
        sign = "+" if v >= 0 else ""
        col  = "#00d4a0" if v >= 0 else "#f56565"
        return '<span style="color:' + col + ';font-family:monospace;font-size:11px;font-weight:600;">' + sign + "${:,.0f}".format(v) + "</span>"

    pnl_badge = ""
    if has_pnl:
        pnl_sign  = "+" if pnl_val >= 0 else ""
        pnl_color = "#00d4a0" if pnl_val >= 0 else "#f56565"
        pnl_bg    = "rgba(0,212,160,0.10)" if pnl_val >= 0 else "rgba(245,101,101,0.10)"
        pnl_border= "rgba(0,212,160,0.30)" if pnl_val >= 0 else "rgba(245,101,101,0.30)"
        pnl_label = "UNREALISED P&L"
        pnl_badge = (
            '<div style="display:inline-flex;align-items:center;gap:10px;'
            'background:' + pnl_bg + ';border:1px solid ' + pnl_border + ';'
            'border-radius:4px;padding:6px 14px;margin-bottom:14px;">'
            '<span style="font-size:9px;color:#4a5568;font-family:monospace;'
            'text-transform:uppercase;letter-spacing:0.1em;">' + pnl_label + '</span>'
            '<span style="font-family:monospace;font-size:20px;font-weight:600;color:' + pnl_color + ';">'
            + pnl_sign + "${:,.0f}".format(pnl_val) + '</span>'
            '<span style="font-size:10px;color:#4a5568;font-family:monospace;">'
            'Leg A: ' + fmt_pnl(pnl_a_val) + '  Leg B: ' + fmt_pnl(pnl_b_val)
            + '</span>'
            '</div>'
        )
    dir_label   = ("LONG  — Buy the Spread" if is_long else "SHORT  — Sell the Spread")
    opt_type    = "Bull Call Spread" if is_long else "Bear Put Spread"
    opt_action  = "Buy Call + Sell Call (higher strike)" if is_long else "Buy Put + Sell Put (lower strike)"
    opt_logic   = (
        sig["a"] + " expected to rise back to fair value." if is_long
        else sig["a"] + " expected to fall back to fair value."
    )

    progress_bar = (
        '<div style="height:5px;background:rgba(255,255,255,0.07);border-radius:3px;margin-top:6px;">'
        '<div style="width:' + str(pct_clamped) + '%;height:100%;background:' + accent + ';border-radius:3px;'
        'transition:width 0.3s;"></div>'
        '</div>'
        '<div style="display:flex;justify-content:space-between;margin-top:4px;">'
        '<span style="font-size:9px;color:#4a5568;font-family:monospace;">Entry: ' + entry_z_str + '</span>'
        '<span style="font-size:9px;color:' + accent + ';font-family:monospace;font-weight:600;">'
        + pct_str + ' of the way to target</span>'
        '<span style="font-size:9px;color:#4a5568;font-family:monospace;">Target: 0.0</span>'
        '</div>'
    )

    html = (
        '<div style="background:' + bg + ';border:1px solid ' + border + ';'
        'border-top:3px solid ' + accent + ';padding:20px;border-radius:6px;margin-bottom:20px;">'

        '<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px;">'
        '<div>'
        '<h2 style="margin:0 0 4px;color:' + accent + ';font-family:monospace;font-size:22px;">'
        + sig["a"] + ' <span style="color:#2d3748;font-weight:300;">/</span> ' + sig["b"] + '</h2>'
        '<span style="font-size:12px;color:#8892a4;font-family:monospace;">' + dir_label + '</span>'
        '</div>'
        '<div style="text-align:right;">'
        '<p style="margin:0 0 2px;font-size:10px;color:#4a5568;font-family:monospace;letter-spacing:0.1em;">Z-SCORE NOW</p>'
        '<p style="margin:0;font-family:monospace;font-size:30px;font-weight:600;color:' + accent + ';">' + z_str + '</p>'
        '<p style="margin:2px 0 0;font-size:10px;color:#4a5568;font-family:monospace;">hedge ratio β = ' + beta_str + '</p>'
        '</div>'
        '</div>'

        '<div style="background:rgba(0,0,0,0.2);border-radius:4px;padding:10px 14px;margin-bottom:14px;">'
        '<p style="margin:0 0 2px;font-size:10px;color:#4a5568;font-family:monospace;'
        'text-transform:uppercase;letter-spacing:0.1em;">Trade Progress — Distance to Profit Target (0.0)</p>'
        + progress_bar +
        '</div>'

        + pnl_badge
        + '<div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;">'

        '<div style="background:rgba(0,0,0,0.25);border-radius:4px;padding:12px 14px;'
        'border:1px solid rgba(255,255,255,0.05);">'
        '<p style="margin:0 0 8px;font-size:10px;color:#4a9eff;font-family:monospace;'
        'text-transform:uppercase;letter-spacing:0.1em;">Option 1 — Stocks (Beta-Weighted)</p>'
        + (pnl_badge if has_pnl else '')
        + _row(leg1_action + "  " + sig["a"],
               sha_str + " shs"
               + ("  entry " + epa_str + "  →  now " + pa_str if epa_str else "  @ " + pa_str)
               + "  =  " + not_a_str,
               leg1_col)
        + _row(leg2_action + "  " + sig["b"],
               shb_str + " shs"
               + ("  entry " + epb_str + "  →  now " + pb_str if epb_str else "  @ " + pb_str)
               + "  =  " + not_b_str,
               leg2_col)
        + _row("Total Deployed", total_str + "  of $" + str(STARTING_CAPITAL), "#e8eaf0")
        + _row("Share Ratio (β-neutral)",
               legs["ratio"] + "  (β=" + str(round(abs(sig["beta"]), 2)) + ")",
               "#e8c96d")
        + _row("Risk Imbalance (Δ-neutral check)",
               ("+" if legs["risk_imbalance"] >= 0 else "") + "${:,.2f}".format(legs["risk_imbalance"])
               + ("  ✓ neutral" if abs(legs["risk_imbalance"]) < 20 else "  ⚠ check sizing"),
               "#00d4a0" if abs(legs["risk_imbalance"]) < 20 else "#f5a623")
        + _row("Dollar Imbalance",
               ("+" if legs["imbalance"] >= 0 else "") + "${:,.2f}".format(legs["imbalance"])
               + ("  ✓ balanced" if legs["dollar_imbal"] < 20 else "  rebalance"),
               "#00d4a0" if legs["dollar_imbal"] < 20 else "#f5a623")
        + '</div>'

        '<div style="background:rgba(0,0,0,0.25);border-radius:4px;padding:12px 14px;'
        'border:1px solid rgba(255,255,255,0.05);">'
        '<p style="margin:0 0 8px;font-size:10px;color:#a78bfa;font-family:monospace;'
        'text-transform:uppercase;letter-spacing:0.1em;">Option 2 — Options Equivalent</p>'
        + _row("Strategy", opt_type, "#a78bfa")
        + _row("Structure", opt_action, "#e8eaf0")
        + '<div style="margin-top:8px;padding-top:6px;border-top:1px solid rgba(255,255,255,0.05);">'
        '<p style="margin:0 0 4px;font-size:11px;color:#8892a4;line-height:1.5;">' + opt_logic + '</p>'
        '<p style="margin:0;font-size:11px;color:#f5a623;">Max loss capped at stop ±' + str(STOP_Z) + '</p>'
        '</div>'
        '</div>'

        '</div></div>'
    )

    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# 6. PAIR CHART
# =============================================================================
def render_pair_chart(p):
    z_data     = p["z_series"]
    open_trade = p.get("open_trade")
    is_long    = (open_trade or {}).get("direction") == "LONG"
    is_short   = (open_trade or {}).get("direction") == "SHORT"

    fig = go.Figure()

    # Active trade zone shading
    if open_trade:
        ez          = float(open_trade["entry_z"])
        y0          = min(ez, EXIT_Z)
        y1          = max(ez, EXIT_Z)
        shade_color = "rgba(0,255,204,0.10)" if is_long else "rgba(255,75,75,0.10)"
        border_col  = "rgba(0,255,204,0.35)" if is_long else "rgba(255,75,75,0.35)"
        fig.add_hrect(y0=y0, y1=y1, fillcolor=shade_color,
                      line=dict(color=border_col, width=1, dash="dot"), layer="below")

    # Z-score line
    fig.add_trace(go.Scatter(
        x=z_data.index, y=z_data,
        line=dict(color="#00d1ff", width=2.5),
        name="Z-Score", opacity=0.92,
        hovertemplate="%{x|%b %d %Y}  Z = %{y:.2f}<extra></extra>",
    ))

    # Glowing active entry marker
    if open_trade and open_trade.get("entry_date") in z_data.index:
        glow_col  = "#00ffcc" if is_long else "#ff4b4b"
        entry_dt  = open_trade["entry_date"]
        entry_y   = float(open_trade["entry_z"])
        glow_bg_o = "rgba(0,255,204,0.15)" if is_long else "rgba(255,75,75,0.15)"
        glow_bg_m = "rgba(0,255,204,0.30)" if is_long else "rgba(255,75,75,0.30)"
        glow_sym  = "triangle-up" if is_long else "triangle-down"

        fig.add_trace(go.Scatter(x=[entry_dt], y=[entry_y], mode="markers",
            marker=dict(color=glow_bg_o, size=32, symbol="circle"),
            showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[entry_dt], y=[entry_y], mode="markers",
            marker=dict(color=glow_bg_m, size=21, symbol="circle"),
            showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[entry_dt], y=[entry_y], mode="markers",
            marker=dict(color=glow_col, size=14, symbol=glow_sym,
                        line=dict(color="#0b0e14", width=2)),
            name="Active Entry",
            hovertemplate=(
                "<b>ACTIVE ENTRY (" + (open_trade["direction"] or "") + ")</b><br>"
                + entry_dt.strftime("%b %d, %Y")
                + "  Z at entry: " + str(round(entry_y, 2))
                + "<br>Z now: " + str(round(p["curr_z"], 2))
                + "<extra></extra>"
            ),
        ))
        fig.add_vline(x=entry_dt.isoformat(),
                      line=dict(color=glow_col, width=1.2, dash="dot"))

    # Threshold lines
    fig.add_hline(y=EXIT_Z, line=dict(color="#ffffff", width=1.2, dash="dot"),
                  annotation_text="TARGET / EXIT (0.0)",
                  annotation_font=dict(color="#8892a4", size=9),
                  annotation_position="bottom right")
    fig.add_hline(y=ENTRY_Z, line=dict(color="#ff4b4b", width=1.5),
                  annotation_text="SHORT ENTRY (+2.25)",
                  annotation_font=dict(color="#ff4b4b", size=9),
                  annotation_position="top left")
    fig.add_hline(y=-ENTRY_Z, line=dict(color="#00ffcc", width=1.5),
                  annotation_text="LONG ENTRY (-2.25)",
                  annotation_font=dict(color="#00ffcc", size=9),
                  annotation_position="bottom left")
    fig.add_hline(y=STOP_Z, line=dict(color="#f5a623", width=2.0, dash="dash"),
                  annotation_text="STOP LOSS (+3.5)",
                  annotation_font=dict(color="#f5a623", size=9),
                  annotation_position="top right")
    fig.add_hline(y=-STOP_Z, line=dict(color="#f5a623", width=2.0, dash="dash"),
                  annotation_text="STOP LOSS (-3.5)",
                  annotation_font=dict(color="#f5a623", size=9),
                  annotation_position="bottom right")

    # Axis ranges
    last_date    = z_data.index[-1]
    one_year_ago = last_date - pd.DateOffset(years=1)
    x_start      = max(one_year_ago, z_data.index[0])
    x_pad        = (last_date - x_start) * 0.12
    x_end        = last_date + x_pad
    visible_z    = z_data[z_data.index >= x_start]
    y_min        = float(visible_z.min())
    y_max        = float(visible_z.max())
    y_pad        = max((y_max - y_min) * 0.15, 0.5)
    y_lo         = min(y_min - y_pad, -STOP_Z - 0.4)
    y_hi         = max(y_max + y_pad,  STOP_Z + 0.4)

    # Status annotation
    annotations = []
    if open_trade:
        s_color  = "#00ffcc" if is_long else "#ff4b4b"
        pct_val  = float(open_trade.get("pct_to_target", 0))
        ann_txt  = (
            "<b>OPEN — " + str(open_trade["direction"]) + "</b><br>"
            "Z now: " + str(round(p["curr_z"], 2)) + "  Target: 0.0<br>"
            + str(round(pct_val, 0)) + "% of the way there"
        )
        annotations.append(dict(
            x=x_end, y=p["curr_z"], xref="x", yref="y",
            text=ann_txt, showarrow=True, arrowhead=2, arrowsize=0.9,
            arrowcolor=s_color, ax=-95, ay=0,
            font=dict(family="IBM Plex Mono", size=10, color=s_color),
            bgcolor="rgba(11,14,20,0.92)",
            bordercolor=s_color, borderwidth=1, borderpad=6, align="left",
        ))
    else:
        annotations.append(dict(
            x=x_end, y=p["curr_z"], xref="x", yref="y",
            text="NEUTRAL<br>Watching for ±" + str(ENTRY_Z),
            showarrow=False,
            font=dict(family="IBM Plex Mono", size=9, color="#4a5568"),
            bgcolor="rgba(11,14,20,0.85)",
            bordercolor="#1e2330", borderwidth=1, borderpad=5,
        ))

    fig.update_layout(
        template="plotly_dark", height=400,
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
                    dict(count=3,  label="3M", step="month", stepmode="backward"),
                    dict(count=6,  label="6M", step="month", stepmode="backward"),
                    dict(count=1,  label="1Y", step="year",  stepmode="backward"),
                    dict(step="all", label="All"),
                ],
            ),
        ),
        yaxis=dict(range=[y_lo, y_hi], showgrid=False, zeroline=False, autorange=False),
        hovermode="x unified",
        font=dict(family="IBM Plex Mono"),
    )

    return fig


# =============================================================================
# 7. CHART FOOTER
# =============================================================================
def render_chart_footer(p):
    ot          = p.get("open_trade")
    coint_color = "#00ffcc" if p["is_cointegrated"] else "#f5a623"
    coint_label = "COINTEGRATED" if p["is_cointegrated"] else "DRIFTING"
    beta_str    = str(round(float(p["beta"]), 2))
    z_str       = str(round(float(p["curr_z"]), 2))
    adf_str     = "p=" + str(round(float(p["adf_pval"]), 3))

    if ot and ot.get("entry_date"):
        ot_color  = "#00ffcc" if ot["direction"] == "LONG" else "#ff4b4b"
        try:
            hold_days = (datetime.now().date() - ot["entry_date"].date()).days
        except Exception:
            hold_days = 0
        status_html = (
            '<span style="color:' + ot_color + ';font-family:monospace;font-size:11px;">'
            + ot["direction"] + ' OPEN — ' + str(hold_days) + 'd held'
            '</span>'
        )
    else:
        status_html = '<span style="color:#2d3748;font-family:monospace;font-size:11px;">Neutral</span>'

    st.markdown(
        '<div style="display:flex;justify-content:space-between;align-items:center;'
        'font-size:11px;margin-top:-12px;padding:0 6px 14px;">'
        '<span style="font-family:monospace;color:#4a5568;">β = <b style="color:#e8eaf0;">' + beta_str + '</b></span>'
        '<span style="font-family:monospace;color:#4a5568;">Z = <b style="color:#00d1ff;">' + z_str + '</b></span>'
        '<span style="font-family:monospace;color:' + coint_color + ';">' + coint_label + ' (' + adf_str + ')</span>'
        + status_html +
        '</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# 8. CHART LEGEND
# =============================================================================
def render_chart_legend():
    st.markdown(
        '<div style="display:flex;flex-wrap:wrap;gap:16px;padding:10px 4px 16px;'
        'font-family:monospace;font-size:11px;color:#4a5568;'
        'border-bottom:1px solid rgba(255,255,255,0.05);margin-bottom:16px;">'
        '<span><span style="color:#00ffcc;">──</span>  Long entry threshold</span>'
        '<span><span style="color:#ff4b4b;">──</span>  Short entry threshold</span>'
        '<span><span style="color:#ffffff;opacity:0.4;">┄┄</span>  Exit / profit target (0.0)</span>'
        '<span><span style="color:#f5a623;">─ ─</span>  Stop loss (±3.5)</span>'
        '<span><span style="color:#00ffcc;font-size:14px;">▲</span>/<span style="color:#ff4b4b;font-size:14px;">▼</span>  Active entry (glowing = position open)</span>'
        '<span><span style="background:rgba(0,255,204,0.15);padding:1px 6px;">■</span>  Teal zone = profit room left (Long)</span>'
        '<span><span style="background:rgba(255,75,75,0.15);padding:1px 6px;">■</span>  Red zone = profit room left (Short)</span>'
        '</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# 9. ENGINE EXPLAINER
# =============================================================================
def render_engine_explainer():
    st.markdown(
        '<div style="background:#111318;border-radius:6px;padding:22px 26px;'
        'border-left:3px solid #4a9eff;">'
        '<p style="font-family:monospace;font-size:12px;color:#4a9eff;margin:0 0 16px;'
        'text-transform:uppercase;letter-spacing:0.12em;">How the Engine Works — State Machine Hysteresis</p>'
        '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:22px;">'

        '<div>'
        '<p style="font-family:monospace;font-size:12px;color:#e8eaf0;margin:0 0 6px;">① Entry — Switch ON</p>'
        '<p style="font-size:12px;color:#8892a4;line-height:1.7;margin:0;">'
        'When Z crosses <b style="color:#e8eaf0;">±2.25</b> for the first time, '
        'a position opens. All subsequent re-crosses of the same threshold are '
        '<b style="color:#f5a623;">ignored</b>. '
        'No double-down. No averaging-in. One trade per pair at a time.'
        '</p></div>'

        '<div>'
        '<p style="font-family:monospace;font-size:12px;color:#e8eaf0;margin:0 0 6px;">② Noise Phase — Hold</p>'
        '<p style="font-size:12px;color:#8892a4;line-height:1.7;margin:0;">'
        'Z may bounce around the threshold — normal market noise. '
        'The shaded zone on each chart shows remaining profit room. '
        'The progress bar on the signal card shows how far the trade has moved '
        'toward the <b style="color:#ffffff;">0.0 target</b>.'
        '</p></div>'

        '<div>'
        '<p style="font-family:monospace;font-size:12px;color:#e8eaf0;margin:0 0 6px;">③ Exit — Switch OFF</p>'
        '<p style="font-size:12px;color:#8892a4;line-height:1.7;margin:0;">'
        'Position closes when Z crosses <b style="color:#ffffff;">0.0</b> '
        '(mean reversion complete — full profit) or hits '
        '<b style="color:#f5a623;">±3.5</b> (stop loss activated). '
        'Only after exit does the engine reset and scan for the next entry.'
        '</p></div>'

        '</div></div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# 10. MAIN TERMINAL
# =============================================================================
def main():
    col_title, col_meta = st.columns([3, 1])
    with col_title:
        st.title("Omni-Arb Terminal v5.7")
        st.caption(
            "Universe: S&P 500 Sector Pairs  |  Engine: State-Machine Cointegration  |  "
            "Capital: $" + str(STARTING_CAPITAL) + " per signal  |  "
            "Refreshed: " + datetime.now().strftime("%b %d %Y %H:%M ET")
        )
    with col_meta:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div style="text-align:right;font-family:monospace;font-size:10px;color:#4a5568;line-height:1.8;">'
            'Entry threshold  ±' + str(ENTRY_Z) + '<br>'
            'Stop loss  ±' + str(STOP_Z) + '<br>'
            'Profit target  0.0<br>'
            'Rolling window  ' + str(ROLLING_WINDOW) + 'd'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    with st.spinner("Syncing market data..."):
        data_raw = get_market_data()

    all_pairs, active_pairs = process_pairs(data_raw)

    n_active  = len(active_pairs)
    n_neutral = len(all_pairs) - n_active
    s_color   = "#00ffcc" if n_active > 0 else "#4a5568"

    st.markdown(
        '<div style="display:flex;align-items:center;gap:16px;margin-bottom:18px;">'
        '<h2 style="margin:0;font-family:monospace;">Active Signals</h2>'
        '<span style="font-family:monospace;font-size:12px;padding:4px 12px;border-radius:3px;'
        'color:' + s_color + ';border:1px solid ' + s_color + ';">'
        + str(n_active) + ' open  /  ' + str(n_neutral) + ' neutral'
        '</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    if active_pairs:
        for sig in active_pairs:
            render_trade_card(sig)
    else:
        st.markdown(
            '<div style="background:#111318;border:1px dashed rgba(255,255,255,0.08);'
            'border-radius:6px;padding:32px;text-align:center;">'
            '<p style="font-family:monospace;font-size:13px;color:#4a5568;margin:0 0 8px;">'
            'No active deviations — all pairs within ±' + str(ENTRY_Z) + 'σ'
            '</p>'
            '<p style="font-family:monospace;font-size:11px;color:#2d3748;margin:0;">'
            'Engine is live. Signals appear the moment Z crosses the entry threshold.'
            '</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    st.markdown(
        '<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:8px;">'
        '<h2 style="margin:0;font-family:monospace;">Pair Analysis</h2>'
        '<span style="font-family:monospace;font-size:11px;color:#4a5568;">'
        '1-year default view — use range buttons to zoom in/out'
        '</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    render_chart_legend()

    chart_cols = st.columns(2)
    for i, p in enumerate(all_pairs):
        with chart_cols[i % 2]:
            ot     = p.get("open_trade")
            accent = (
                "#00ffcc" if (ot and ot["direction"] == "LONG") else
                "#ff4b4b" if (ot and ot["direction"] == "SHORT") else
                "#4a5568"
            )
            badge = (
                '<span style="font-family:monospace;font-size:9px;font-weight:600;'
                'letter-spacing:0.12em;padding:2px 8px;border-radius:2px;'
                'color:' + accent + ';border:1px solid ' + accent + ';opacity:0.85;">'
                + ot["direction"] + ' OPEN</span>'
                if ot else
                '<span style="font-family:monospace;font-size:9px;color:#2d3748;">NEUTRAL</span>'
            )
            st.markdown(
                '<div style="display:flex;justify-content:space-between;'
                'align-items:center;margin-bottom:6px;">'
                '<p style="margin:0;font-family:monospace;font-size:15px;font-weight:500;'
                'color:' + accent + ';">'
                + p["a"] + ' <span style="color:#2d3748;font-weight:300;">/</span> ' + p["b"]
                + '</p>' + badge + '</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(render_pair_chart(p), use_container_width=True)
            render_chart_footer(p)

    st.divider()
    render_engine_explainer()


if __name__ == "__main__":
    main()

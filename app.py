# =============================================================================
# OMNI-ARB v5.5  |  Statistical Arbitrage Terminal
# State-machine logic: Entry Hysteresis / Toggle Logic
# Focus: Live Signals & Active Trade Tracking
# =============================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go

# =============================================================================
# 0. CONFIG & THEME
# =============================================================================
st.set_page_config(page_title="Omni-Arb v5.5", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .main { background-color: #0b0e14; color: #e1e1e1; }
    .stPlotlyChart { background-color: #0b0e14; border-radius: 10px; }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.5px; }
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
# 2. SIZING  (live signal cards)
# =============================================================================
def compute_legs(sig, capital=STARTING_CAPITAL):
    target_per_leg = capital / 2
    shares_a = max(1, int(target_per_leg / sig["price_a"]))
    shares_b = max(1, int(shares_a * sig["beta"]))
    return {
        "shares_a":   shares_a,
        "notional_a": shares_a * sig["price_a"],
        "shares_b":   shares_b,
        "notional_b": shares_b * sig["price_b"],
        "total_cost": shares_a * sig["price_a"] + shares_b * sig["price_b"],
    }

# =============================================================================
# 3. DATA ENGINE (Includes State-Machine Replay)
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
        )
        curr_z = z_series.iloc[-1]

        long_entries  = z_series[(z_series <= -ENTRY_Z) & (z_series.shift(1) > -ENTRY_Z)]
        short_entries = z_series[(z_series >= ENTRY_Z)  & (z_series.shift(1) < ENTRY_Z)]
        exits = z_series[
            ((z_series >= 0) & (z_series.shift(1) < 0))
            | ((z_series <= 0) & (z_series.shift(1) > 0))
        ]

        adf_pval = adfuller(spread)[1]

        # ── Open trade detection (state-machine replay) ──────────────────
        # Walk the full z_series to find if the most recent entry cross
        # is still unclosed (no zero-cross or stop hit since then).
        in_open         = False
        open_direction  = None
        open_entry_date = None
        open_entry_z    = None

        for _i in range(1, len(z_series)):
            _zp = z_series.iloc[_i - 1]
            _zc = z_series.iloc[_i]
            _dt = z_series.index[_i]

            if not in_open:
                if _zp > -ENTRY_Z and _zc <= -ENTRY_Z:
                    in_open = True; open_direction = "LONG"
                    open_entry_date = _dt; open_entry_z = _zc
                elif _zp < ENTRY_Z and _zc >= ENTRY_Z:
                    in_open = True; open_direction = "SHORT"
                    open_entry_date = _dt; open_entry_z = _zc
            else:
                closed = (
                    (open_direction == "LONG"  and _zp < 0 and _zc >= 0) or
                    (open_direction == "SHORT" and _zp > 0 and _zc <= 0) or
                    abs(_zc) >= STOP_Z
                )
                if closed:
                    in_open = False; open_direction = None
                    open_entry_date = None; open_entry_z = None

        open_trade = (
            {"direction": open_direction, "entry_date": open_entry_date,
             "entry_z": open_entry_z, "curr_z": curr_z}
            if in_open else None
        )

        # Direction: at-threshold OR currently in an open trade
        direction_now = (
            "LONG"    if curr_z <= -ENTRY_Z else
            "SHORT"   if curr_z >= ENTRY_Z  else
            (open_direction if open_trade else "NEUTRAL")
        )

        info = {
            "pair":          f"{ticker_a}/{ticker_b}",
            "a":             ticker_a,
            "b":             ticker_b,
            "price_a":       pair_df[ticker_a].iloc[-1],
            "price_b":       pair_df[ticker_b].iloc[-1],
            "curr_z":        curr_z,
            "beta":          betas.iloc[-1],
            "z_series":      z_series,
            "spread":        spread,
            "pair_df":       pair_df,
            "long_spots":    long_entries,
            "short_spots":   short_entries,
            "exit_spots":    exits,
            "is_cointegrated": adf_pval < 0.05,
            "open_trade":    open_trade,
            "direction":     direction_now,
        }
        processed.append(info)
        if info["direction"] != "NEUTRAL":
            active.append(info)
    return processed, active


# =============================================================================
# 4. UI HELPERS
# =============================================================================
def render_trade_card(sig):
    is_long     = sig["direction"] == "LONG"
    accent      = "#00ffcc" if is_long else "#ff4b4b"
    bg          = "rgba(0, 255, 204, 0.05)" if is_long else "rgba(255, 75, 75, 0.05)"
    legs        = compute_legs(sig)
    leg1_verb   = "BUY"            if is_long else "SELL"
    leg2_verb   = "SELL"           if is_long else "BUY"
    leg1_col    = "#00ffcc"        if is_long else "#ff4b4b"
    leg2_col    = "#ff4b4b"        if is_long else "#00ffcc"
    action_text = "BUY THE SPREAD" if is_long else "SELL THE SPREAD"
    state_text  = "Lagging"        if is_long else "Overextended"
    z_str       = f"{sig['curr_z']:.2f}"

    st.markdown(
        '<div style="background:' + bg + ';border:1px solid ' + accent + ';padding:20px;'
        'border-radius:8px;margin-bottom:20px;">'
        '<div style="display:flex;justify-content:space-between;align-items:center;">'
        '<h2 style="margin:0;color:' + accent + ';font-size:24px;font-family:monospace;">'
        + sig["a"] + " / " + sig["b"] + "</h2>"
        '<div style="text-align:right;">'
        '<span style="font-size:12px;color:#8892a4;">CURRENT Z-SCORE</span><br>'
        '<b style="font-size:20px;color:' + accent + ';">' + z_str + "</b>"
        "</div></div>"
        '<hr style="border:0;border-top:1px solid rgba(255,255,255,0.1);margin:15px 0;">'
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">'
        "<div>"
        '<p style="font-size:11px;color:#4a5568;margin-bottom:8px;text-transform:uppercase;">Action Required</p>'
        '<b style="font-size:16px;color:#fff;">' + action_text + "</b><br>"
        '<span style="font-size:13px;color:#8892a4;">' + sig["a"] + " is " + state_text + "</span>"
        "</div>"
        '<div style="background:rgba(0,0,0,0.2);padding:10px;border-radius:4px;">'
        '<p style="font-size:10px;color:#4a5568;margin:0;">EXECUTION GUIDE ($1K)</p>'
        '<div style="display:flex;justify-content:space-between;margin-top:5px;font-family:monospace;">'
        '<span style="color:' + leg1_col + ';">' + leg1_verb + " " + str(legs["shares_a"]) + " " + sig["a"] + "</span>"
        '<span style="color:' + leg2_col + ';">' + leg2_verb + " " + str(legs["shares_b"]) + " " + sig["b"] + "</span>"
        "</div></div></div></div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# 5. MAIN DASHBOARD
# =============================================================================
def main():
    st.title("📟 Omni-Arb Terminal v5.5")
    st.caption(
        f"Asset Universe: S&P 500 Pairs  |  Engine: State-Machine Cointegration  |  "
        f"Live Capital: ${STARTING_CAPITAL}"
    )
    st.divider()

    with st.spinner("Fetching market data..."):
        data_raw = get_market_data()

    all_pairs, active_pairs = process_pairs(data_raw)

    # ── Active signals ────────────────────────────────────────
    if active_pairs:
        st.subheader("⚠️ ACTIVE TRADE SIGNALS")
        for sig in active_pairs:
            render_trade_card(sig)
    else:
        st.info("✨ No active deviations detected. Monitoring market for entry...")

    st.divider()

    # ── Pair analysis charts ──────────────────────────────────
    st.subheader("📊 PAIR ANALYSIS")
    chart_cols = st.columns(2)

    for i, p in enumerate(all_pairs):
        with chart_cols[i % 2]:
            z_data     = p["z_series"].dropna()
            open_trade = p.get("open_trade")
            is_long    = (open_trade or {}).get("direction") == "LONG"
            is_short   = (open_trade or {}).get("direction") == "SHORT"

            st.markdown(f"<h4 style='text-align: center; color: #8892a4;'>{p['pair']}</h4>", unsafe_allow_html=True)

            fig = go.Figure()

            # ── Active trade zone shading ─────────────────────
            # Shades from entry_z toward 0.0 to show remaining profit room
            if open_trade:
                entry_z_val  = open_trade["entry_z"]
                if is_long:
                    shade_y_lo = min(entry_z_val, EXIT_Z)
                    shade_y_hi = EXIT_Z
                else:
                    shade_y_lo = EXIT_Z
                    shade_y_hi = max(entry_z_val, EXIT_Z)
                
                # Add shaded rectangle for active trade
                fig.add_hrect(
                    y0=shade_y_lo, y1=shade_y_hi, 
                    fillcolor="rgba(0, 255, 204, 0.1)" if is_long else "rgba(255, 75, 75, 0.1)",
                    layer="below", line_width=0,
                )

                # Add floating "OPEN TRADE" banner to chart
                banner_color = "#00ffcc" if is_long else "#ff4b4b"
                fig.add_annotation(
                    text=f"STATUS: OPEN {open_trade['direction']}",
                    xref="paper", yref="paper", x=0.5, y=0.95, showarrow=False,
                    font=dict(size=12, color="#000", family="monospace"),
                    bgcolor=banner_color, borderpad=4, bordercolor=banner_color
                )

            # Draw Z-Score Line
            fig.add_trace(go.Scatter(
                x=z_data.index, y=z_data, mode='lines', 
                line=dict(color='#00d1ff', width=1.5), name='Z-Score'
            ))

            # Threshold Lines
            fig.add_hline(y=EXIT_Z, line=dict(color="#ffffff", width=1, dash="dot"))
            fig.add_hline(y=ENTRY_Z, line=dict(color="#ff4b4b", width=1))
            fig.add_hline(y=-ENTRY_Z, line=dict(color="#00ffcc", width=1))
            fig.add_hline(y=STOP_Z, line=dict(color="#f5a623", width=2, dash="dash"))
            fig.add_hline(y=-STOP_Z, line=dict(color="#f5a623", width=2, dash="dash"))

            fig.update_layout(
                template="plotly_dark", height=250, margin=dict(l=10, r=10, t=10, b=20),
                paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14", showlegend=False,
                xaxis=dict(range=[z_data.index[-100], z_data.index[-1] + pd.Timedelta(days=5)], showgrid=False),
                yaxis=dict(range=[-4, 4], showgrid=False, zeroline=False)
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── State-machine explainer (Enhanced and moved to main) ───────────────────────────────
    st.divider()
    st.markdown(
        '<div style="background:#111318;border:1px solid rgba(255,255,255,0.07);'
        'border-left:3px solid #4a9eff;border-radius:4px;padding:20px;">'
        '<h3 style="margin:0 0 15px;font-family:monospace;font-size:16px;color:#4a9eff;'
        'text-transform:uppercase;letter-spacing:0.1em;">Engine Methodology: The State-Machine</h3>'
        '<p style="color:#8892a4; font-size:13px; margin-bottom: 20px;">'
        'This terminal operates on strict <b>Entry Hysteresis (Toggle) Logic</b>. This acts like a physical circuit breaker to protect your capital from market noise and over-trading.</p>'
        '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:20px;">'
        '<div><p style="margin:0 0 8px;font-size:14px;color:#00ffcc;font-family:monospace;">① The Entry Phase</p>'
        '<p style="margin:0;font-size:13px;color:#e1e1e1;line-height:1.6;">'
        'Z crosses ±2.25. Position opens. Switch flips <b>ON</b>. '
        'All subsequent re-crosses are completely ignored. There is no double-down, and no averaging-in.</p></div>'
        '<div><p style="margin:0 0 8px;font-size:14px;color:#f5a623;font-family:monospace;">② The Noise Phase</p>'
        '<p style="margin:0;font-size:13px;color:#e1e1e1;line-height:1.6;">'
        'The Z-score wiggles above and below the threshold line. The system does nothing. '
        'Your capital is protected from whip-sawing. Opportunity cost: your allocated cash is locked until the exit triggers.</p></div>'
        '<div><p style="margin:0 0 8px;font-size:14px;color:#ffffff;font-family:monospace;">③ The Exit Phase</p>'
        '<p style="margin:0;font-size:13px;color:#e1e1e1;line-height:1.6;">'
        'Z returns to exactly 0.0. Both legs automatically close, P&L is realized, and the switch resets to <b>NEUTRAL</b>. '
        'A hard stop-loss at ±3.5 cuts losses early if the spread structurally breaks down.</p></div>'
        "</div></div>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()

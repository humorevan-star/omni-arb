# =============================================================================
# OMNI-ARB v5.6  |  Statistical Arbitrage Terminal
# State-machine logic: Entry Hysteresis / Toggle Logic
# Focus: Execution-Centric Signals (Stocks & Options)
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
st.set_page_config(page_title="Omni-Arb v5.6", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .main { background-color: #0b0e14; color: #e1e1e1; }
    .stPlotlyChart { background-color: #0b0e14; border-radius: 10px; }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.5px; }
    code { color: #e1e1e1 !important; background: rgba(255,255,255,0.1) !important; }
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
# 2. SIZING & EXECUTION LOGIC
# =============================================================================
def compute_legs(sig, capital=STARTING_CAPITAL):
    target_per_leg = capital / 2
    shares_a = max(1, int(target_per_leg / sig["price_a"]))
    shares_b = max(1, int(shares_a * sig["beta"]))
    return {
        "shares_a":   shares_a,
        "shares_b":   shares_b,
        "total_cost": (shares_a * sig["price_a"]) + (shares_b * sig["price_b"]),
    }

# =============================================================================
# 3. DATA ENGINE (State-Machine Processor)
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
        
        curr_z = z_series.iloc[-1]
        adf_pval = adfuller(spread)[1]

        # ── State-Machine Replay (The "Toggle" Logic) ──────────────────
        in_open         = False
        open_direction  = None
        open_entry_z    = None

        for _i in range(1, len(z_series)):
            _zp = z_series.iloc[_i - 1]
            _zc = z_series.iloc[_i]

            if not in_open:
                if _zp > -ENTRY_Z and _zc <= -ENTRY_Z:
                    in_open = True; open_direction = "LONG"; open_entry_z = _zc
                elif _zp < ENTRY_Z and _zc >= ENTRY_Z:
                    in_open = True; open_direction = "SHORT"; open_entry_z = _zc
            else:
                closed = (
                    (open_direction == "LONG"  and _zp < 0 and _zc >= 0) or
                    (open_direction == "SHORT" and _zp > 0 and _zc <= 0) or
                    abs(_zc) >= STOP_Z
                )
                if closed:
                    in_open = False; open_direction = None

        open_trade = (
            {"direction": open_direction, "entry_z": open_entry_z, "curr_z": curr_z}
            if in_open else None
        )

        direction_now = open_direction if open_trade else "NEUTRAL"

        info = {
            "pair":          f"{ticker_a}/{ticker_b}",
            "a":             ticker_a,
            "b":             ticker_b,
            "price_a":       pair_df[ticker_a].iloc[-1],
            "price_b":       pair_df[ticker_b].iloc[-1],
            "curr_z":        curr_z,
            "beta":          betas.iloc[-1],
            "z_series":      z_series,
            "is_cointegrated": adf_pval < 0.05,
            "open_trade":    open_trade,
            "direction":     direction_now,
        }
        processed.append(info)
        if info["direction"] != "NEUTRAL":
            active.append(info)
    return processed, active


# =============================================================================
# 4. UI COMPONENTS
# =============================================================================
def render_trade_card(sig):
    is_long     = sig["direction"] == "LONG"
    accent      = "#00ffcc" if is_long else "#ff4b4b"
    bg          = "rgba(0, 255, 204, 0.05)" if is_long else "rgba(255, 75, 75, 0.05)"
    legs        = compute_legs(sig)
    
    # Textual logic for the specific request
    action_verb = "BUY THE SPREAD" if is_long else "SELL THE SPREAD"
    leg1_action = "BUY" if is_long else "SELL"
    leg2_action = "SELL" if is_long else "BUY"
    state_text  = "is Lagging" if is_long else "is Overextended"
    
    # Options strategy mapping
    opt_strat = "BULL CALL SPREAD / SELL PUTS" if is_long else "BEAR PUT SPREAD / SELL CALLS"

    st.markdown(
        f"""
        <div style="background:{bg}; border:1px solid {accent}; padding:20px; border-radius:8px; margin-bottom:20px;">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h2 style="margin:0; color:{accent}; font-family:monospace;">{sig['a']} / {sig['b']}</h2>
                <div style="text-align:right;">
                    <span style="font-size:12px; color:#8892a4;">CURRENT Z-SCORE</span><br>
                    <b style="font-size:20px; color:{accent};">{sig['curr_z']:.2f}</b>
                </div>
            </div>
            
            <hr style="border:0; border-top:1px solid rgba(255,255,255,0.1); margin:15px 0;">
            
            <div style="display:grid; grid-template-columns: 1.2fr 1fr; gap:20px;">
                <div>
                    <p style="font-size:11px; color:#4a5568; margin-bottom:4px; text-transform:uppercase;">Action Required</p>
                    <b style="font-size:18px; color:#fff;">{action_verb}</b><br>
                    <span style="font-size:14px; color:{accent}; font-weight:bold;">
                        {leg1_action} {sig['a']} + {leg2_action} {sig['b']}
                    </span><br>
                    <span style="font-size:12px; color:#8892a4;">{sig['a']} {state_text} relative to {sig['b']}</span>
                </div>
                
                <div style="background:rgba(0,0,0,0.3); padding:12px; border-radius:4px; border:1px solid rgba(255,255,255,0.05);">
                    <p style="font-size:10px; color:#8892a4; margin:0 0 8px 0; letter-spacing:1px; font-family:monospace;">EXECUTION GUIDE ($1K CAPITAL)</p>
                    
                    <div style="margin-bottom:12px;">
                        <span style="font-size:10px; color:#4a9eff; font-weight:bold;">OPTION 1 (STOCKS):</span><br>
                        <code style="font-size:12px; font-family:monospace;">
                            {leg1_action} {legs['shares_a']} {sig['a']} / {leg2_action} {legs['shares_b']} {sig['b']}
                        </code>
                    </div>
                    
                    <div>
                        <span style="font-size:10px; color:#a78bfa; font-weight:bold;">OPTION 2 (DERIVATIVES):</span><br>
                        <span style="font-size:11px; color:#e1e1e1; font-family:monospace;">{opt_strat} on {sig['a']}</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =============================================================================
# 5. MAIN TERMINAL
# =============================================================================
def main():
    st.title("📟 Omni-Arb Terminal v5.6")
    st.caption(f"Universe: S&P 500 Pairs | Mode: Live Signal Tracking | Static Capital: ${STARTING_CAPITAL}")
    st.divider()

    with st.spinner("Syncing with Market Data..."):
        data_raw = get_market_data()

    all_pairs, active_pairs = process_pairs(data_raw)

    # ── ACTIVE SIGNALS ────────────────────────────────────────
    if active_pairs:
        st.subheader("⚠️ ACTIVE TRADE SIGNALS")
        for sig in active_pairs:
            render_trade_card(sig)
    else:
        st.info("✨ No active deviations currently maintained by the state-machine.")

    st.divider()

    # ── PAIR MONITORING ───────────────────────────────────────
    st.subheader("📊 PAIR ANALYSIS")
    chart_cols = st.columns(2)

    for i, p in enumerate(all_pairs):
        with chart_cols[i % 2]:
            z_data     = p["z_series"]
            open_trade = p.get("open_trade")
            
            st.markdown(f"<p style='text-align: center; color: #8892a4; margin-bottom:-10px; font-family:monospace;'>{p['pair']}</p>", unsafe_allow_html=True)

            fig = go.Figure()

            # Active Trade Highlight
            if open_trade:
                color = "rgba(0, 255, 204, 0.1)" if open_trade["direction"] == "LONG" else "rgba(255, 75, 75, 0.1)"
                fig.add_hrect(y0=open_trade["entry_z"], y1=EXIT_Z, fillcolor=color, line_width=0, layer="below")
                fig.add_annotation(text="ACTIVE", xref="paper", yref="paper", x=0.5, y=0.9, showarrow=False, 
                                   font=dict(color="#000"), bgcolor="#00ffcc" if open_trade["direction"] == "LONG" else "#ff4b4b")

            fig.add_trace(go.Scatter(x=z_data.index, y=z_data, line=dict(color='#00d1ff', width=1.5)))
            fig.add_hline(y=EXIT_Z, line=dict(color="#ffffff", width=1, dash="dot"))
            fig.add_hline(y=ENTRY_Z, line=dict(color="#ff4b4b", width=1))
            fig.add_hline(y=-ENTRY_Z, line=dict(color="#00ffcc", width=1))

            fig.update_layout(
                template="plotly_dark", height=240, margin=dict(l=10, r=10, t=20, b=10),
                paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14", showlegend=False,
                xaxis=dict(range=[z_data.index[-90], z_data.index[-1]], showgrid=False),
                yaxis=dict(range=[-4, 4], showgrid=False, zeroline=False)
            )
            st.plotly_chart(fig, use_container_width=True)

    # ── STATE-MACHINE LOGIC EXPLAINER ─────────────────────────
    st.divider()
    st.markdown(
        """
        <div style="background:#111318; border-radius:4px; padding:20px; border-left:4px solid #4a9eff;">
            <p style="font-family:monospace; font-size:14px; color:#4a9eff; margin-bottom:10px;">ENGINE LOGIC: STATE-MACHINE HYSTERESIS</p>
            <div style="display:grid; grid-template-columns: repeat(3, 1fr); gap:20px; font-size:12px; color:#8892a4;">
                <div><b style="color:#eee;">1. Entry Toggle</b><br>Cross ±2.25 flips the 'ON' switch. Re-crosses are ignored to prevent over-trading.</div>
                <div><b style="color:#eee;">2. Noise Phase</b><br>Capital stays locked while Z-score vibrates. This avoids unnecessary fees and slippage.</div>
                <div><b style="color:#eee;">3. Reset Trigger</b><br>Positions exit ONLY at 0.0 or ±3.5 (Stop). Only then does the 'ON' switch reset to Neutral.</div>
            </div>
        </div>
        """, unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

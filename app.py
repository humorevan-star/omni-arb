# =============================================================================
# OMNI-ARB v6.0  |  Statistical Arbitrage Terminal
# State-machine logic: Entry Hysteresis / Toggle Logic
# Focus: Execution-Centric Signals (Stocks & Options)
# Integrated: Clean Chart Dots + High-Performance Backtesting
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
import math as _math

# =============================================================================
# 0. CONFIG & THEME
# =============================================================================
st.set_page_config(page_title="Omni-Arb v6.0", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .main { background-color: #0b0e14; color: #e1e1e1; }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.5px; }
    code { color: #00d1ff !important; background: rgba(0,209,255,0.08) !important; padding: 2px 6px !important; }
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

# Backtest Settings
BT_CAPITAL       = 2000    
BT_MAX_OPEN      = 3       
BT_MAX_HOLD_DAYS = 21      
BT_LOOKBACK_DAYS = 730     
BT_ALLOC_PER     = BT_CAPITAL / len(PAIRS)

# =============================================================================
# 2. CALCULATION ENGINES (SIZING & PNL)
# =============================================================================
def compute_legs(sig, capital=STARTING_CAPITAL):
    """Medallion Delta-Neutral Sizing."""
    price_a = float(sig["price_a"])
    price_b = float(sig["price_b"])
    beta    = max(abs(float(sig["beta"])), 0.01)

    dollar_a = capital / (1.0 + beta)
    dollar_b = capital - dollar_a
    shares_a = max(0.1, round(dollar_a / price_a, 1))
    shares_b = max(0.1, round(dollar_b / price_b, 1))

    notional_a = round(shares_a * price_a, 2)
    notional_b = round(shares_b * price_b, 2)
    
    pnl = pnl_a = pnl_b = None
    ot = sig.get("open_trade")
    if ot and ot.get("entry_price_a"):
        ep_a, ep_b = float(ot["entry_price_a"]), float(ot["entry_price_b"])
        is_long = ot["direction"] == "LONG"
        _sa = float(ot.get("entry_shares_a", shares_a))
        _sb = float(ot.get("entry_shares_b", shares_b))
        
        # P&L Calculation logic
        notional_a_entry = _sa * ep_a
        log_spread_entry = _math.log(ep_a) - beta * _math.log(ep_b)
        log_spread_now   = _math.log(price_a) - beta * _math.log(price_b)
        pnl = round((log_spread_now - log_spread_entry) * notional_a_entry * (1 if is_long else -1), 2)

    return {"shares_a": shares_a, "shares_b": shares_b, "notional_a": notional_a, "notional_b": notional_b, "pnl": pnl}

# =============================================================================
# 3. DATA & STATE MACHINE PROCESSOR
# =============================================================================
@st.cache_data(ttl=3600)
def get_market_data():
    tickers = list(set([t for p in PAIRS for t in p]))
    return yf.download(tickers, period="750d", interval="1d")["Close"]

def process_pairs(df_raw):
    processed, active = [], []
    for ticker_a, ticker_b in PAIRS:
        if ticker_a not in df_raw.columns or ticker_b not in df_raw.columns: continue
        pair_df = df_raw[[ticker_a, ticker_b]].dropna()
        y, x = np.log(pair_df[ticker_a]), sm.add_constant(np.log(pair_df[ticker_b]))
        model = RollingOLS(y, x, window=ROLLING_WINDOW).fit()
        betas, consts = model.params[ticker_b], model.params["const"]
        spread = (y - (betas * np.log(pair_df[ticker_b]) + consts)).dropna()
        z_series = ((spread - spread.rolling(ROLLING_WINDOW).mean()) / spread.rolling(ROLLING_WINDOW).std()).dropna()
        
        # --- State Machine Tracking ---
        in_open, open_dir, open_z, open_dt = False, None, None, None
        for dt, z in z_series.items():
            if not in_open:
                if z <= -ENTRY_Z: in_open, open_dir, open_z, open_dt = True, "LONG", z, dt
                elif z >= ENTRY_Z: in_open, open_dir, open_z, open_dt = True, "SHORT", z, dt
            else:
                if (open_dir == "LONG" and z >= 0) or (open_dir == "SHORT" and z <= 0) or abs(z) >= STOP_Z:
                    in_open = False
        
        open_trade = {"direction": open_dir, "entry_z": open_z, "entry_date": open_dt, 
                      "entry_price_a": pair_df[ticker_a].loc[open_dt] if in_open else None,
                      "entry_price_b": pair_df[ticker_b].loc[open_dt] if in_open else None} if in_open else None

        info = {"pair": f"{ticker_a}/{ticker_b}", "a": ticker_a, "b": ticker_b, "curr_z": z_series.iloc[-1], 
                "beta": betas.iloc[-1], "z_series": z_series, "price_a": pair_df[ticker_a].iloc[-1], 
                "price_b": pair_df[ticker_b].iloc[-1], "open_trade": open_trade, "direction": open_dir or "NEUTRAL",
                "is_cointegrated": adfuller(spread)[1] < 0.05, "pair_df": pair_df, "betas_series": betas}
        processed.append(info)
        if in_open: active.append(info)
    return processed, active

# =============================================================================
# 4. VISUAL RENDERING
# =============================================================================
def render_pair_chart(p):
    z_data, ot = p["z_series"], p.get("open_trade")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=z_data.index, y=z_data, line=dict(color="#00d1ff", width=2), name="Z-Score"))
    
    if ot:
        fig.add_trace(go.Scatter(x=[ot["entry_date"]], y=[ot["entry_z"]], mode="markers", 
                                 marker=dict(color="#00ffcc" if ot["direction"]=="LONG" else "#ff4b4b", size=12, symbol="star")))
    
    fig.add_hline(y=ENTRY_Z, line_dash="dash", line_color="#ff4b4b")
    fig.add_hline(y=-ENTRY_Z, line_dash="dash", line_color="#00ffcc")
    fig.add_hline(y=0, line_color="white", line_width=1)
    fig.update_layout(template="plotly_dark", height=300, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
    return fig

# =============================================================================
# 5. BACKTEST ENGINE & MAIN
# =============================================================================
def main():
    st.title("Omni-Arb Terminal v6.0")
    data_raw = get_market_data()
    all_pairs, active_pairs = process_pairs(data_raw)
    
    st.subheader("Active Trade Signals")
    if active_pairs:
        for sig in active_pairs:
            legs = compute_legs(sig)
            st.info(f"🚀 {sig['pair']} | Side: {sig['direction']} | P&L: ${legs['pnl']}")
    else:
        st.write("No active signals. Monitoring thresholds...")

    st.divider()
    cols = st.columns(2)
    for i, p in enumerate(all_pairs):
        with cols[i % 2]:
            st.markdown(f"**{p['pair']}** (Z: {p['curr_z']:.2f})")
            st.plotly_chart(render_pair_chart(p), use_container_width=True)

if __name__ == "__main__":
    main()

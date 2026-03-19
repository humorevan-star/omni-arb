import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
from datetime import datetime

# ==========================================
# 0. CONFIG & THEME
# ==========================================
st.set_page_config(page_title="Omni-Arb Command Center", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #fafafa; }
    h1, h2, h3, p { font-family: 'Inter', sans-serif; }
    .stPlotlyChart { margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 1. PARAMETERS & SIZING ENGINE
# ==========================================
PAIRS = [('XOM', 'CVX'), ('V', 'MA'), ('NVDA', 'AMD'), ('KO', 'PEP'), ('GOOGL', 'META')]
ENTRY_Z, EXIT_Z, STOP_Z = 2.25, 0.0, 3.5
ROLLING_WINDOW = 60
STARTING_CAPITAL = 1000 

def compute_legs(sig, capital=STARTING_CAPITAL):
    target_per_leg = capital / 2
    shares_a = int(target_per_leg / sig["price_a"])
    if shares_a == 0: shares_a = 1
    actual_notional_a = shares_a * sig["price_a"]
    
    shares_b = int(shares_a * sig["beta"])
    if shares_b == 0: shares_b = 1
    actual_notional_b = shares_b * sig["price_b"]
    
    return {
        "shares_a": shares_a,
        "notional_a": actual_notional_a,
        "shares_b": shares_b,
        "notional_b": actual_notional_b,
        "total_cost": actual_notional_a + actual_notional_b
    }

# ==========================================
# 2. DATA ENGINE (SIGNAL DETECTION)
# ==========================================
@st.cache_data(ttl=3600)
def get_market_data():
    tickers = list(set([t for p in PAIRS for t in p]))
    data = yf.download(tickers, period="250d", interval="1d")['Close']
    return data

def process_pairs(df_raw):
    processed = []
    active = []
    
    for ticker_a, ticker_b in PAIRS:
        if ticker_a not in df_raw or ticker_b not in df_raw: continue
        
        pair_df = df_raw[[ticker_a, ticker_b]].dropna()
        y, x = np.log(pair_df[ticker_a]), sm.add_constant(np.log(pair_df[ticker_b]))
        
        model = RollingOLS(y, x, window=ROLLING_WINDOW).fit()
        betas = model.params[ticker_b]
        consts = model.params['const']
        
        spread = y - (betas * np.log(pair_df[ticker_b]) + consts)
        spread = spread.dropna()
        
        adf_pval = adfuller(spread)[1]
        z_series = (spread - spread.rolling(ROLLING_WINDOW).mean()) / spread.rolling(ROLLING_WINDOW).std()
        curr_z = z_series.iloc[-1]
        
        # --- HISTORICAL SIGNAL DETECTION ---
        # Find points where Z crosses thresholds for "Spots"
        long_entries = z_series[(z_series <= -ENTRY_Z) & (z_series.shift(1) > -ENTRY_Z)]
        short_entries = z_series[(z_series >= ENTRY_Z) & (z_series.shift(1) < ENTRY_Z)]
        
        # Exit is when it crosses 0 (fair value)
        exits = z_series[((z_series >= 0) & (z_series.shift(1) < 0)) | ((z_series <= 0) & (z_series.shift(1) > 0))]
        
        info = {
            'pair': f"{ticker_a}/{ticker_b}",
            'a': ticker_a, 'b': ticker_b,
            'price_a': pair_df[ticker_a].iloc[-1],
            'price_b': pair_df[ticker_b].iloc[-1],
            'curr_z': curr_z,
            'beta': betas.iloc[-1],
            'z_series': z_series,
            'long_spots': long_entries,
            'short_spots': short_entries,
            'exit_spots': exits,
            'is_cointegrated': adf_pval < 0.05,
            'adf_pval': adf_pval,
            'direction': "LONG" if curr_z <= -ENTRY_Z else "SHORT" if curr_z >= ENTRY_Z else "NEUTRAL"
        }
        
        processed.append(info)
        if info['direction'] != "NEUTRAL":
            active.append(info)
            
    return processed, active

# ==========================================
# 3. UI RENDERING
# ==========================================
def render_trade_card(sig):
    is_long = sig["direction"] == "LONG"
    accent = "#00d4a0" if is_long else "#f56565"
    bg_accent = "rgba(0,212,160,0.08)" if is_long else "rgba(245,101,101,0.07)"
    verb_a, verb_b = ("BUY", "SELL") if is_long else ("SELL", "BUY")
    legs = compute_legs(sig)

    html = f"""
    <div style="background:{bg_accent}; border-left: 5px solid {accent}; padding: 20px; border-radius: 4px; margin-bottom: 25px; color: white;">
        <div style="display: flex; justify-content: space-between;">
            <div>
                <h2 style="margin:0;">{sig['a']} vs {sig['b']}</h2>
                <p style="margin: 4px 0; color: {accent}; font-weight: bold;">{sig['direction']} SIGNAL | Z: {sig['curr_z']:.2f}</p>
            </div>
        </div>
        <div style="margin-top: 15px; background: rgba(0,0,0,0.2); padding: 12px; border-radius: 4px;">
            <p style="margin: 0 0 8px 0; font-size: 10px; color: #4a5568; font-weight: bold;">BROKER INSTRUCTIONS ($1K BALANCE)</p>
            <div style="display: flex; justify-content: space-between;">
                <span><b>{verb_a}</b> {legs['shares_a']} shs {sig['a']} @ ${sig['price_a']:.2f}</span>
                <span><b>{verb_b}</b> {legs['shares_b']} shs {sig['b']} @ ${sig['price_b']:.2f}</span>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ==========================================
# 4. MAIN APP
# ==========================================
st.title("🚀 Omni-Arb Command Center v5.1")
st.caption(f"Portfolio Balance: ${STARTING_CAPITAL} | Tracking Cointegrated Pairs")
st.divider()

data_raw = get_market_data()
all_pairs, active_pairs = process_pairs(data_raw)

# --- ACTIVE SIGNALS ---
st.subheader("⚡ Active Execution Signals")
if not active_pairs:
    st.info("Searching for entry points... All pairs are currently in neutral territory.")
else:
    for sig in active_pairs:
        render_trade_card(sig)

st.divider()

# --- CHARTS WITH SPOTS ---
st.subheader("📊 Market Analysis & Historical Signals")
cols = st.columns(2)
for i, p in enumerate(all_pairs):
    with cols[i % 2]:
        st.markdown(f"#### {p['pair']}")
        
        z_data = p['z_series'].dropna()
        fig = go.Figure()
        
        # Main Z-Line
        fig.add_trace(go.Scatter(x=z_data.index, y=z_data, line=dict(color='#4a5568', width=1.5), name="Z-Score", opacity=0.6))
        
        # 🟢 LONG ENTRY SPOTS
        fig.add_trace(go.Scatter(x=p['long_spots'].index, y=p['long_spots'], mode='markers', 
                                 marker=dict(color='#00d4a0', size=10, symbol='circle'), name="Buy Entry"))
        
        # 🔴 SHORT ENTRY SPOTS
        fig.add_trace(go.Scatter(x=p['short_spots'].index, y=p['short_spots'], mode='markers', 
                                 marker=dict(color='#f56565', size=10, symbol='circle'), name="Sell Entry"))
        
        # ⚪ EXIT SPOTS
        fig.add_trace(go.Scatter(x=p['exit_spots'].index, y=p['exit_spots'], mode='markers', 
                                 marker=dict(color='white', size=7, symbol='diamond'), name="Exit (0.0)"))

        # Threshold Lines
        fig.add_hline(y=ENTRY_Z, line_dash="dot", line_color="#f56565", opacity=0.3)
        fig.add_hline(y=-ENTRY_Z, line_dash="dot", line_color="#00d4a0", opacity=0.3)
        fig.add_hline(y=0, line_color="white", opacity=0.2)
        
        last_date = z_data.index[-1]
        x_padding = (last_date - z_data.index[0]) * 0.12 # Breathing room
        
        fig.update_layout(
            template="plotly_dark", height=320, showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(range=[z_data.index[0], last_date + x_padding], showgrid=False),
            yaxis=dict(range=[-4, 4], zeroline=False)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"**Current Z:** {p['curr_z']:.2f} | **Beta:** {p['beta']:.2f} | **Coint:** {'Healthy' if p['is_cointegrated'] else 'Drift'}")

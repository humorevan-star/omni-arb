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
# 0. CONFIG & THEME (REFINED)
# ==========================================
st.set_page_config(page_title="Omni-Arb v5.2", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .main { background-color: #0b0e14; color: #e1e1e1; }
    .stPlotlyChart { background-color: #0b0e14; border-radius: 10px; }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    h1, h2, h3 { font-family: 'Inter', sans-serif; letter-spacing: -0.5px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 1. PARAMETERS & SIZING (STRICT $1K)
# ==========================================
PAIRS = [('XOM', 'CVX'), ('V', 'MA'), ('NVDA', 'AMD'), ('KO', 'PEP'), ('GOOGL', 'META')]
ENTRY_Z = 2.25
STOP_Z = 3.5
EXIT_Z = 0.0
ROLLING_WINDOW = 60
STARTING_CAPITAL = 1000 

def compute_legs(sig, capital=STARTING_CAPITAL):
    target_per_leg = capital / 2
    shares_a = max(1, int(target_per_leg / sig["price_a"]))
    actual_notional_a = shares_a * sig["price_a"]
    
    shares_b = max(1, int(shares_a * sig["beta"]))
    actual_notional_b = shares_b * sig["price_b"]
    
    return {
        "shares_a": shares_a, "notional_a": actual_notional_a,
        "shares_b": shares_b, "notional_b": actual_notional_b,
        "total_cost": actual_notional_a + actual_notional_b
    }

# ==========================================
# 2. DATA ENGINE
# ==========================================
@st.cache_data(ttl=3600)
def get_market_data():
    tickers = list(set([t for p in PAIRS for t in p]))
    data = yf.download(tickers, period="250d", interval="1d")['Close']
    return data

def process_pairs(df_raw):
    processed, active = [], []
    for ticker_a, ticker_b in PAIRS:
        if ticker_a not in df_raw or ticker_b not in df_raw: continue
        
        pair_df = df_raw[[ticker_a, ticker_b]].dropna()
        y, x = np.log(pair_df[ticker_a]), sm.add_constant(np.log(pair_df[ticker_b]))
        
        model = RollingOLS(y, x, window=ROLLING_WINDOW).fit()
        betas, consts = model.params[ticker_b], model.params['const']
        spread = y - (betas * np.log(pair_df[ticker_b]) + consts)
        spread = spread.dropna()
        
        z_series = (spread - spread.rolling(ROLLING_WINDOW).mean()) / spread.rolling(ROLLING_WINDOW).std()
        curr_z = z_series.iloc[-1]
        
        # Spot Detection
        long_entries = z_series[(z_series <= -ENTRY_Z) & (z_series.shift(1) > -ENTRY_Z)]
        short_entries = z_series[(z_series >= ENTRY_Z) & (z_series.shift(1) < ENTRY_Z)]
        exits = z_series[((z_series >= 0) & (z_series.shift(1) < 0)) | ((z_series <= 0) & (z_series.shift(1) > 0))]
        
        info = {
            'pair': f"{ticker_a}/{ticker_b}", 'a': ticker_a, 'b': ticker_b,
            'price_a': pair_df[ticker_a].iloc[-1], 'price_b': pair_df[ticker_b].iloc[-1],
            'curr_z': curr_z, 'beta': betas.iloc[-1], 'z_series': z_series,
            'long_spots': long_entries, 'short_spots': short_entries, 'exit_spots': exits,
            'is_cointegrated': adfuller(spread)[1] < 0.05,
            'direction': "LONG" if curr_z <= -ENTRY_Z else "SHORT" if curr_z >= ENTRY_Z else "NEUTRAL"
        }
        processed.append(info)
        if info['direction'] != "NEUTRAL": active.append(info)
    return processed, active

# ==========================================
# 3. UI RENDERING
# ==========================================
def render_trade_card(sig):
    is_long = sig["direction"] == "LONG"
    accent = "#00ffcc" if is_long else "#ff4b4b"
    bg = "rgba(0, 255, 204, 0.05)" if is_long else "rgba(255, 75, 75, 0.05)"
    legs = compute_legs(sig)
    
    st.markdown(f"""
    <div style="background:{bg}; border: 1px solid {accent}; padding: 20px; border-radius: 8px; margin-bottom: 20px;">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <h2 style="margin:0; color:{accent}; font-size: 24px;">{sig['a']} / {sig['b']}</h2>
            <div style="text-align: right;">
                <span style="font-size: 12px; color: #8892a4;">CURRENT Z-SCORE</span><br>
                <b style="font-size: 20px; color: {accent};">{sig['curr_z']:.2f}</b>
            </div>
        </div>
        <hr style="border: 0; border-top: 1px solid rgba(255,255,255,0.1); margin: 15px 0;">
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
            <div>
                <p style="font-size: 11px; color: #4a5568; margin-bottom: 8px; text-transform: uppercase;">Action Required</p>
                <b style="font-size: 16px; color: #fff;">{'BUY' if is_long else 'SELL'} THE SPREAD</b><br>
                <span style="font-size: 13px; color: #8892a4;">{sig['a']} is {"Lagging" if is_long else "Overextended"}</span>
            </div>
            <div style="background: rgba(0,0,0,0.2); padding: 10px; border-radius: 4px;">
                <p style="font-size: 10px; color: #4a5568; margin: 0;">EXECUTION GUIDE ($1K)</p>
                <div style="display: flex; justify-content: space-between; margin-top: 5px; font-family: monospace;">
                    <span style="color:{accent};">{'BUY' if is_long else 'SELL'} {legs['shares_a']} {sig['a']}</span>
                    <span style="color:{'#ff4b4b' if is_long else '#00ffcc'};">{'SELL' if is_long else 'BUY'} {legs['shares_b']} {sig['b']}</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 4. MAIN DASHBOARD
# ==========================================
st.title("📟 Omni-Arb Terminal v5.2")
st.caption(f"Asset Universe: S&P 500 Pairs | Engine: Dynamic Cointegration | Capital: ${STARTING_CAPITAL}")
st.divider()

data_raw = get_market_data()
all_pairs, active_pairs = process_pairs(data_raw)

# --- TOP: ACTIVE ALERTS ---
if active_pairs:
    st.subheader("⚠️ ACTIVE TRADE SIGNALS")
    for sig in active_pairs:
        render_trade_card(sig)
else:
    st.write("✨ No active deviations detected. Monitoring market for entry...")

st.divider()

# --- BOTTOM: BEAUTIFIED CHARTS ---
st.subheader("📊 PAIR ANALYSIS")
cols = st.columns(2)
for i, p in enumerate(all_pairs):
    with cols[i % 2]:
        z_data = p['z_series'].dropna()
        fig = go.Figure()
        
        # Historical Z-Score Line
        fig.add_trace(go.Scatter(x=z_data.index, y=z_data, line=dict(color='#00d1ff', width=1.5), name="Spread Z", opacity=0.8))
        
        # Markers
        fig.add_trace(go.Scatter(x=p['long_spots'].index, y=p['long_spots'], mode='markers', marker=dict(color='#00ffcc', size=10), name="ENTRY (BUY)"))
        fig.add_trace(go.Scatter(x=p['short_spots'].index, y=p['short_spots'], mode='markers', marker=dict(color='#ff4b4b', size=10), name="ENTRY (SELL)"))
        fig.add_trace(go.Scatter(x=p['exit_spots'].index, y=p['exit_spots'], mode='markers', marker=dict(color='#ffffff', size=6, symbol='diamond'), name="EXIT"))

        # --- THE BEAUTIFIED THRESHOLDS ---
        # 0 (Exit)
        fig.add_hline(y=EXIT_Z, line=dict(color="#ffffff", width=1, dash="dot"), annotation_text="EXIT (0.0)", annotation_position="right")
        # Entry Lines
        fig.add_hline(y=ENTRY_Z, line=dict(color="#ff4b4b", width=1), annotation_text="SELL ENTRY (+2.25)", annotation_position="top left")
        fig.add_hline(y=-ENTRY_Z, line=dict(color="#00ffcc", width=1), annotation_text="BUY ENTRY (-2.25)", annotation_position="bottom left")
        # Stop Loss
        fig.add_hline(y=STOP_Z, line=dict(color="#f5a623", width=2, dash="dash"), annotation_text="STOP LOSS (+3.5)", annotation_position="top right")
        fig.add_hline(y=-STOP_Z, line=dict(color="#f5a623", width=2, dash="dash"), annotation_text="STOP LOSS (-3.5)", annotation_position="bottom right")

        # Layout & Padding
        last_date = z_data.index[-1]
        x_range_end = last_date + (last_date - z_data.index[0]) * 0.15 # 15% padding
        
        fig.update_layout(
            template="plotly_dark", height=380, margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
            showlegend=False,
            xaxis=dict(range=[z_data.index[0], x_range_end], showgrid=False),
            yaxis=dict(range=[-4, 4], showgrid=False, zeroline=False)
        )
        
        st.markdown(f"### {p['a']} vs {p['b']}")
        st.plotly_chart(fig, use_container_width=True)
        
        # Footer Stat Bar
        st.markdown(f"""
            <div style="display:flex; justify-content:space-between; font-size:12px; color:#4a5568; margin-top:-15px; padding: 0 10px;">
                <span>Beta: <b>{p['beta']:.2f}</b></span>
                <span>Cointegration: <b style="color:{'#00ffcc' if p['is_cointegrated'] else '#f5a623'};">{'HEALTHY' if p['is_cointegrated'] else 'DRIFTING'}</b></span>
            </div><br>
        """, unsafe_allow_html=True)

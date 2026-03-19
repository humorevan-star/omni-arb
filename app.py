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
    .stPlotlyChart { margin-bottom: 40px; }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 1. PARAMETERS & SIZING ENGINE
# ==========================================
PAIRS = [('XOM', 'CVX'), ('V', 'MA'), ('NVDA', 'AMD'), ('KO', 'PEP'), ('GOOGL', 'META')]
ENTRY_Z, EXIT_Z, STOP_Z = 2.25, 0.0, 3.5
ROLLING_WINDOW = 60
STARTING_CAPITAL = 1000  # Adjusted for your request

def compute_legs(sig, capital=STARTING_CAPITAL):
    """Calculates dollar-neutral sizing for a retail budget."""
    target_per_leg = capital / 2
    
    # Leg A
    shares_a = int(target_per_leg / sig["price_a"])
    if shares_a == 0: shares_a = 1
    actual_notional_a = shares_a * sig["price_a"]
    
    # Leg B (Hedged via Beta)
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
# 2. DATA ENGINE (ROLLING OLS + ADF)
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
        
        # Rolling OLS prevents look-ahead bias
        model = RollingOLS(y, x, window=ROLLING_WINDOW).fit()
        betas = model.params[ticker_b]
        consts = model.params['const']
        
        # Spread calculation: Actual Y - Predicted Y
        spread = y - (betas * np.log(pair_df[ticker_b]) + consts)
        spread = spread.dropna()
        
        # Cointegration Check
        adf_pval = adfuller(spread)[1]
        
        # Z-Score Calculation
        z_score_series = (spread - spread.rolling(ROLLING_WINDOW).mean()) / spread.rolling(ROLLING_WINDOW).std()
        curr_z = z_score_series.iloc[-1]
        
        info = {
            'pair': f"{ticker_a}/{ticker_b}",
            'a': ticker_a, 'b': ticker_b,
            'price_a': pair_df[ticker_a].iloc[-1],
            'price_b': pair_df[ticker_b].iloc[-1],
            'curr_z': curr_z,
            'beta': betas.iloc[-1],
            'z_series': z_score_series,
            'is_cointegrated': adf_pval < 0.05,
            'adf_pval': adf_pval,
            'direction': "LONG" if curr_z <= -ENTRY_Z else "SHORT" if curr_z >= ENTRY_Z else "NEUTRAL"
        }
        
        processed.append(info)
        if info['direction'] != "NEUTRAL":
            active.append(info)
            
    return processed, active

# ==========================================
# 3. UI RENDERING FUNCTIONS
# ==========================================
def render_trade_card(sig):
    is_long = sig["direction"] == "LONG"
    accent = "#00d4a0" if is_long else "#f56565"
    bg_accent = "rgba(0,212,160,0.08)" if is_long else "rgba(245,101,101,0.07)"
    
    z_meaning = "LOW (Cheap)" if is_long else "HIGH (Expensive)"
    action_text = "BUY THE SPREAD" if is_long else "SELL THE SPREAD"
    
    legs = compute_legs(sig)
    verb_a = "BUY" if is_long else "SELL"
    verb_b = "SELL" if is_long else "BUY"

    html = f"""
    <div style="background:{bg_accent}; border-left: 5px solid {accent}; padding: 20px; border-radius: 4px; margin-bottom: 25px; color: white;">
        <div style="display: flex; justify-content: space-between;">
            <div>
                <h2 style="margin:0;">{sig['a']} vs {sig['b']}</h2>
                <p style="margin: 4px 0; color: {accent}; font-weight: bold;">{action_text} | Z-Score: {sig['curr_z']:.2f}</p>
            </div>
            <div style="text-align: right;">
                <p style="margin:0; font-size: 10px; color: #8892a4;">CONDITION</p>
                <p style="margin:0; font-size: 14px; font-weight: bold;">{z_meaning}</p>
            </div>
        </div>
        <div style="margin-top: 15px; display: flex; gap: 40px;">
            <div style="flex: 1; background: rgba(0,0,0,0.2); padding: 12px; border-radius: 4px;">
                <p style="margin: 0 0 8px 0; font-size: 10px; color: #4a5568; font-weight: bold;">EXECUTION GUIDE</p>
                <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                    <span style="color:{accent}; font-weight:bold;">{verb_a}</span>
                    <span>{legs['shares_a']} shs of {sig['a']}</span>
                    <span style="color:#8892a4;">@ ${sig['price_a']:.2f}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color:{'#f56565' if is_long else '#00d4a0'}; font-weight:bold;">{verb_b}</span>
                    <span>{legs['shares_b']} shs of {sig['b']}</span>
                    <span style="color:#8892a4;">@ ${sig['price_b']:.2f}</span>
                </div>
            </div>
            <div style="width: 150px;">
                <p style="margin: 0 0 5px 0; font-size: 10px; color: #4a5568;">TOTAL COST</p>
                <p style="margin:0; font-size: 18px; font-weight: bold;">${legs['total_cost']:,.2f}</p>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# ==========================================
# 4. MAIN APP FLOW
# ==========================================
st.title("🚀 Omni-Arb Command Center v5.0")
st.caption(f"Portfolio Balance: ${STARTING_CAPITAL} | Strategy: Dollar-Neutral Mean Reversion")
st.divider()

data_raw = get_market_data()
all_pairs, active_pairs = process_pairs(data_raw)

# --- SECTION: ACTIVE SIGNALS ---
st.subheader("⚡ Active Execution Signals")
if not active_pairs:
    st.info("No active signals. All pairs within normal historical range.")
else:
    for sig in active_pairs:
        render_trade_card(sig)

st.divider()

# --- SECTION: DETAILED CHARTS ---
st.subheader("📊 Market Analysis & Z-Scores")
cols = st.columns(2)
for i, p in enumerate(all_pairs):
    with cols[i % 2]:
        st.markdown(f"#### {p['pair']}")
        
        # Plotly logic with "Breathing Room" at the end
        z_data = p['z_series'].dropna()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=z_data.index, y=z_data, line=dict(color='#00d1ff', width=2), name="Z-Score"))
        
        # Guide Lines
        fig.add_hline(y=ENTRY_Z, line_dash="dash", line_color="#f56565")
        fig.add_hline(y=-ENTRY_Z, line_dash="dash", line_color="#00d4a0")
        fig.add_hline(y=0, line_color="white")
        
        # X-Axis Padding
        last_date = z_data.index[-1]
        x_padding = (last_date - z_data.index[0]) * 0.1
        
        fig.update_layout(
            template="plotly_dark", height=300, 
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(range=[z_data.index[0], last_date + x_padding], showgrid=False),
            yaxis=dict(range=[-4, 4], zeroline=False)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        coint_color = "green" if p['is_cointegrated'] else "orange"
        st.markdown(f"""
            <div style="font-size: 12px; color: #8892a4; display: flex; justify-content: space-between;">
                <span>Current Z: <b>{p['curr_z']:.2f}</b></span>
                <span>Beta: <b>{p['beta']:.2f}</b></span>
                <span style="color:{coint_color};">Coint: {'PASS' if p['is_cointegrated'] else 'DRIFT'}</span>
            </div>
        """, unsafe_allow_html=True)
        st.write("") # Spacer

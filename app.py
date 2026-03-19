import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go

st.set_page_config(page_title="Omni-Arb Dashboard", layout="wide")
st.title("🚀 Omni-Arb Live Monitor v2.1")

# 1. SETUP
PAIRS = [('V', 'MA'), ('NVDA', 'AMD'), ('KO', 'PEP'), ('XOM', 'CVX'), ('GOOGL', 'META')]
ENTRY_Z, EXIT_Z, STOP_Z = 2.25, 0.0, 3.5

@st.cache_data(ttl=3600)
def get_data():
    tickers = list(set([t for p in PAIRS for t in p]))
    # Fetch extra days to ensure we have enough after cleaning
    data = yf.download(tickers, period="200d", interval="1d")['Close']
    return data

df_raw = get_data()

# 2. DASHBOARD DISPLAY
cols = st.columns(2)

for i, (a, b) in enumerate(PAIRS):
    # --- DATA CLEANING FOR THIS SPECIFIC PAIR ---
    # Pick only the two stocks and drop any days where either is missing
    pair_df = df_raw[[a, b]].dropna()
    
    # Ensure we don't have zeros (which break log math)
    pair_df = pair_df[(pair_df > 0).all(axis=1)]
    
    if len(pair_df) < 60:
        with cols[i % 2]:
            st.warning(f"Not enough data for {a}/{b} yet.")
        continue

    # Math Engine
    y = np.log(pair_df[a])
    x = sm.add_constant(np.log(pair_df[b]))
    
    try:
        model = sm.OLS(y, x).fit()
        beta = model.params[b]
        spread = y - (beta * np.log(pair_df[b]) + model.params['const'])
        z_score = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
        curr_z = z_score.iloc[-1]
        
        # Action Logic
        status = "MONITORING"
        if curr_z < -ENTRY_Z: status = "🟢 LONG SPREAD"
        elif curr_z > ENTRY_Z: status = "🔴 SHORT SPREAD"
        if abs(curr_z) >= STOP_Z: status = "⚠️ STOP LOSS"

        # Plotting
        with cols[i % 2]:
            st.subheader(f"{a} / {b}: {status}")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=z_score.index, y=z_score, name="Z-Score", line=dict(color='cyan')))
            fig.add_hline(y=ENTRY_Z, line_dash="dash", line_color="red")
            fig.add_hline(y=-ENTRY_Z, line_dash="dash", line_color="red")
            fig.add_hline(y=0, line_color="white")
            fig.update_layout(template="plotly_dark", height=300, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)
            st.write(f"**Current Z:** {curr_z:.2f} | **Beta:** {beta:.2f}")
    except Exception as e:
        with cols[i % 2]:
            st.error(f"Error calculating {a}/{b}")

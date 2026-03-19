import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go

st.set_page_config(page_title="Omni-Arb Dashboard", layout="wide")
st.title("🚀 Omni-Arb Live Monitor v2.0")

# 1. SETUP
PAIRS = [('V', 'MA'), ('NVDA', 'AMD'), ('KO', 'PEP'), ('XOM', 'CVX'), ('GOOGL', 'META')]
ENTRY_Z, EXIT_Z, STOP_Z = 2.25, 0.0, 3.5

@st.cache_data(ttl=3600) # Refresh data every hour
def get_data():
    tickers = list(set([t for p in PAIRS for t in p]))
    return yf.download(tickers, period="120d", interval="1d")['Close']

df = get_data()

# 2. DASHBOARD DISPLAY
cols = st.columns(2) # Two columns of charts

for i, (a, b) in enumerate(PAIRS):
    # Math Engine
    y, x = np.log(df[a]), sm.add_constant(np.log(df[b]))
    model = sm.OLS(y, x).fit()
    beta = model.params[b]
    spread = y - (beta * np.log(df[b]) + model.params['const'])
    z_score = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
    curr_z = z_score.iloc[-1]
    
    # Action Logic
    status = "MONITORING"
    color = "white"
    if curr_z < -ENTRY_Z: status, color = "🟢 LONG SPREAD", "green"
    elif curr_z > ENTRY_Z: status, color = "🔴 SHORT SPREAD", "red"
    if abs(curr_z) >= STOP_Z: status, color = "⚠️ STOP LOSS", "orange"

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

# 3. P&L TRACKER (Manual Input)
st.divider()
st.header("💰 Trade Ledger & Cumulative P&L")
st.info("Goal: 50% APY ($1,000 profit on $2,000 bankroll)")
# You can add a dataframe here to track your Tastytrade fills manually

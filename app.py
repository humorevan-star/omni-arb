import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go

st.set_page_config(page_title="Omni-Arb Command Center", layout="wide")
st.title("🚀 Omni-Arb Live Monitor v3.0")

# 1. SETUP & CONFIG
PAIRS = [('V', 'MA'), ('NVDA', 'AMD'), ('KO', 'PEP'), ('XOM', 'CVX'), ('GOOGL', 'META')]
ENTRY_Z, EXIT_Z, STOP_Z = 2.25, 0.0, 3.5

def get_instrument_type(vol):
    if vol > 0.25: return "RATIO BACKSPREAD", "60-70 DTE"
    elif vol > 0.12: return "VERTICAL SPREAD", "30-45 DTE"
    else: return "PURE STOCKS", "N/A"

@st.cache_data(ttl=3600)
def get_data():
    tickers = list(set([t for p in PAIRS for t in p]))
    data = yf.download(tickers, period="200d", interval="1d")['Close']
    return data

df_raw = get_data()

# --- NEW: TOP SUMMARY MATRIX ---
st.subheader("📊 Real-Time Signal Matrix")
summary_data = []

for a, b in PAIRS:
    pair_df = df_raw[[a, b]].dropna()
    if len(pair_df) < 60: continue
    y, x = np.log(pair_df[a]), sm.add_constant(np.log(pair_df[b]))
    model = sm.OLS(y, x).fit()
    spread = y - (model.params[b] * np.log(pair_df[b]) + model.params['const'])
    z_score = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
    curr_z = z_score.iloc[-1]
    
    # Status Logic
    if curr_z > ENTRY_Z: status = "🔴 SHORT"
    elif curr_z < -ENTRY_Z: status = "🟢 LONG"
    elif abs(curr_z) < 0.2: status = "✅ EXIT/NEUTRAL"
    else: status = "⚪ WATCH"
    
    summary_data.append({"Pair": f"{a}/{b}", "Current Z": round(curr_z, 2), "Status": status})

# Display Summary Table
st.table(pd.DataFrame(summary_data))
st.divider()

# --- INDIVIDUAL CHARTS & DETAILED INSTRUCTIONS ---
cols = st.columns(2)
for i, (a, b) in enumerate(PAIRS):
    pair_df = df_raw[[a, b]].dropna()
    y, x = np.log(pair_df[a]), sm.add_constant(np.log(pair_df[b]))
    try:
        model = sm.OLS(y, x).fit()
        beta = model.params[b]
        spread = y - (beta * np.log(pair_df[b]) + model.params['const'])
        z_series = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
        curr_z = z_series.iloc[-1]
        pair_vol = spread.pct_change().std() * np.sqrt(252)

        with cols[i % 2]:
            st.write(f"### {a} vs {b}")
            strat, dte = get_instrument_type(pair_vol)
            
            if abs(curr_z) > ENTRY_Z:
                color = "red" if curr_z > 0 else "green"
                st.markdown(f"**RECOMMENDED:** :{color}[{strat} ({dte})]")
                
                # Instruction Logic
                main_ticker = a
                side = "Puts" if curr_z > 0 else "Calls"
                if strat == "RATIO BACKSPREAD":
                    st.info(f"**{main_ticker} Options:** Sell 1 ATM {side}, Buy 2 OTM {side}")
                elif strat == "VERTICAL SPREAD":
                    st.info(f"**{main_ticker} Options:** Buy 1 ATM {side}, Sell 1 OTM {side}")
            
            # Simplified Chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=z_series.index, y=z_series, line=dict(color='cyan')))
            fig.add_hline(y=ENTRY_Z, line_dash="dash", line_color="red")
            fig.add_hline(y=-ENTRY_Z, line_dash="dash", line_color="red")
            fig.add_hline(y=0, line_color="lime")
            fig.update_layout(template="plotly_dark", height=250, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.divider()
    except: continue

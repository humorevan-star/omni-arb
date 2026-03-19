import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go

st.set_page_config(page_title="Omni-Arb Dashboard", layout="wide")
st.title("🚀 Omni-Arb Live Monitor v2.2")

# 1. SETUP
PAIRS = [('V', 'MA'), ('NVDA', 'AMD'), ('KO', 'PEP'), ('XOM', 'CVX'), ('GOOGL', 'META')]
ENTRY_Z, EXIT_Z, STOP_Z = 2.25, 0.0, 3.5

# --- NEW STRATEGY LOGIC ---
def get_instrument_type(vol):
    if vol > 0.25:
        return "RATIO BACKSPREAD", "High Volatility - Use 1:2 Options"
    elif vol > 0.12:
        return "VERTICAL SPREAD", "Medium Volatility - Standard Options"
    else:
        return "PURE STOCKS", "Low Volatility - Buy/Short Shares Only"

@st.cache_data(ttl=3600)
def get_data():
    tickers = list(set([t for p in PAIRS for t in p]))
    data = yf.download(tickers, period="200d", interval="1d")['Close']
    return data

df_raw = get_data()
cols = st.columns(2)

for i, (a, b) in enumerate(PAIRS):
    pair_df = df_raw[[a, b]].dropna()
    pair_df = pair_df[(pair_df > 0).all(axis=1)]
    
    if len(pair_df) < 60:
        continue

    # Math Engine
    y, x = np.log(pair_df[a]), sm.add_constant(np.log(pair_df[b]))
    try:
        model = sm.OLS(y, x).fit()
        beta = model.params[b]
        spread = y - (beta * np.log(pair_df[b]) + model.params['const'])
        
        # Z-Score Calculation
        z_score_series = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
        curr_z = z_score_series.iloc[-1]
        
        # Volatility Calculation (Standard Deviation of the spread)
        pair_vol = spread.pct_change().std() * np.sqrt(252)

        with cols[i % 2]:
            st.subheader(f"Pair: {a} vs {b}")
            
            # --- THE SIGNAL & STRATEGY BOX ---
            strat_name, strat_desc = get_instrument_type(pair_vol)
            
            if abs(curr_z) > ENTRY_Z:
                color = "red" if curr_z > 0 else "green"
                direction = "SHORT THE SPREAD" if curr_z > 0 else "LONG THE SPREAD"
                st.markdown(f"### :{color}[{direction}]")
                
                # EXECUTION INSTRUCTIONS
                st.success(f"**USE: {strat_name}**\n\n{strat_desc}")
                if strat_name == "RATIO BACKSPREAD":
                    st.info(f"**Action:** Sell 1 ATM {'Put' if curr_z > 0 else 'Call'}, Buy 2 OTM {'Puts' if curr_z > 0 else 'Calls'}")
                else:
                    st.info(f"**Action:** {'Sell' if curr_z > 0 else 'Buy'} {a} / {'Buy' if curr_z > 0 else 'Sell'} {b}")
            else:
                st.info(f"⚪ STATUS: NO TRADE (Z is {curr_z:.2f})")

            # Plotting
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=z_score_series.index, y=z_score_series, name="Z-Score", line=dict(color='cyan')))
            fig.add_hline(y=ENTRY_Z, line_dash="dash", line_color="red")
            fig.add_hline(y=-ENTRY_Z, line_dash="dash", line_color="red")
            fig.add_hline(y=0, line_color="white")
            fig.update_layout(template="plotly_dark", height=250, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption(f"Current Z: {curr_z:.2f} | Hedge Ratio: {beta:.2f} | Ann. Vol: {pair_vol:.1%}")
            st.divider()

    except Exception as e:
        st.error(f"Error on {a}/{b}: {e}")

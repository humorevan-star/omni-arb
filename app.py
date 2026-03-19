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
        z_score = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
        curr_z = z_score.iloc[-1]
        
        with cols[i % 2]:
            st.subheader(f"Pair: {a} vs {b}")
            
            # --- THE SIGNAL BOX ---
            if curr_z > ENTRY_Z:
                st.error(f"🔴 SIGNAL: SHORT THE SPREAD\n\n**Action:** Sell {a} / Buy {b}\n\n**Target:** Exit when Z hits 0.0")
            elif curr_z < -ENTRY_Z:
                st.success(f"🟢 SIGNAL: LONG THE SPREAD\n\n**Action:** Buy {a} / Sell {b}\n\n**Target:** Exit when Z hits 0.0")
            else:
                st.info(f"⚪ STATUS: NO TRADE (Waiting for Z > {ENTRY_Z})")

            # Plotting
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=z_score.index, y=z_score, name="Z-Score", line=dict(color='cyan')))
            fig.add_hline(y=ENTRY_Z, line_dash="dash", line_color="red", annotation_text="ENTRY")
            fig.add_hline(y=-ENTRY_Z, line_dash="dash", line_color="red", annotation_text="ENTRY")
            fig.add_hline(y=0, line_color="white", annotation_text="EXIT (PROFIT)")
            fig.update_layout(template="plotly_dark", height=300, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk Management Info
            st.caption(f"Current Z: {curr_z:.2f} | Hedge Ratio (Beta): {beta:.2f}")
            st.warning(f"Stop Loss: Close trade if Z hits {STOP_Z if curr_z > 0 else -STOP_Z}")
            st.divider()

    except Exception as e:
        st.error(f"Error on {a}/{b}")

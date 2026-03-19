import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go

st.set_page_config(page_title="Omni-Arb Dashboard", layout="wide")
st.title("🚀 Omni-Arb Live Monitor v2.5")

# 1. SETUP
PAIRS = [('V', 'MA'), ('NVDA', 'AMD'), ('KO', 'PEP'), ('XOM', 'CVX'), ('GOOGL', 'META')]
ENTRY_Z, EXIT_Z, STOP_Z = 2.25, 0.0, 3.5

def get_instrument_type(vol):
    if vol > 0.25:
        return "RATIO BACKSPREAD", "High Volatility - Uncapped Upside", "60-70 Days Out"
    elif vol > 0.12:
        return "VERTICAL SPREAD", "Medium Volatility - Capped Risk", "30-45 Days Out"
    else:
        return "PURE STOCKS", "Low Volatility - No Theta Decay", "N/A"

@st.cache_data(ttl=3600)
def get_data():
    tickers = list(set([t for p in PAIRS for t in p]))
    data = yf.download(tickers, period="200d", interval="1d")['Close']
    return data

df_raw = get_data()
cols = st.columns(2)

for i, (a, b) in enumerate(PAIRS):
    pair_df = df_raw[[a, b]].dropna()
    if len(pair_df) < 60: continue

    y, x = np.log(pair_df[a]), sm.add_constant(np.log(pair_df[b]))
    try:
        model = sm.OLS(y, x).fit()
        beta = model.params[b]
        spread = y - (beta * np.log(pair_df[b]) + model.params['const'])
        z_score_series = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
        curr_z = z_score_series.iloc[-1]
        pair_vol = spread.pct_change().std() * np.sqrt(252)

        with cols[i % 2]:
            st.subheader(f"Pair: {a} vs {b}")
            strat_name, strat_desc, dte = get_instrument_type(pair_vol)
            
            if abs(curr_z) > ENTRY_Z:
                color = "red" if curr_z > 0 else "green"
                direction = "SHORT THE SPREAD" if curr_z > 0 else "LONG THE SPREAD"
                st.markdown(f"### :{color}[{direction}]")
                st.success(f"**STRATEGY:** {strat_name} ({dte})")
                
                # SPECIFIC EXECUTION INSTRUCTIONS
                if strat_name == "RATIO BACKSPREAD":
                    # If Z is high, A is expensive (Sell A). If Z is low, A is cheap (Buy A).
                    main_ticker = a
                    opt_type = "Puts" if curr_z > 0 else "Calls"
                    st.info(f"**Trade Options on {main_ticker}:**\n\n"
                            f"1. Sell (1) At-the-Money {opt_type}\n"
                            f"2. Buy (2) Out-of-the-Money {opt_type}\n"
                            f"**Expiration:** {dte}")
                elif strat_name == "VERTICAL SPREAD":
                    st.info(f"**Trade Options on {a}:**\n\n"
                            f"Buy 1 ATM, Sell 1 OTM ({opt_type})\n"
                            f"**Expiration:** {dte}")
                else:
                    st.info(f"**Stock Only:** {'Short' if curr_z > 0 else 'Buy'} {a} vs {'Buy' if curr_z > 0 else 'Short'} {b}")
            else:
                st.info(f"⚪ STATUS: NO TRADE (Z-Score: {curr_z:.2f})")

            # Charting
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=z_score_series.index, y=z_score_series, name="Z-Score", line=dict(color='cyan')))
            fig.add_hline(y=ENTRY_Z, line_dash="dash", line_color="red")
            fig.add_hline(y=-ENTRY_Z, line_dash="dash", line_color="red")
            fig.update_layout(template="plotly_dark", height=230, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"Current Z: {curr_z:.2f} | Vol: {pair_vol:.1%}")
            st.divider()

    except Exception as e:
        st.error(f"Error: {e}")


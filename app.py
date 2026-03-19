import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go

st.set_page_config(page_title="Omni-Arb Command Center", layout="wide")

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    div[data-testid="stMetricValue"] { font-size: 24px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 Omni-Arb Live Monitor v3.5 – TICKER FIXED")

# 1. SETUP - Edit your pairs here
PAIRS = [('XOM', 'CVX'), ('V', 'MA'), ('NVDA', 'AMD'), ('KO', 'PEP'), ('GOOGL', 'META')]
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
    # Downloading 200 days of data for the moving average calculation
    data = yf.download(tickers, period="200d", interval="1d")['Close']
    return data

df_raw = get_data()
cols = st.columns(2)

for i, (ticker_a, ticker_b) in enumerate(PAIRS):
    # Ensure we have data for both tickers
    if ticker_a not in df_raw or ticker_b not in df_raw:
        continue
        
    pair_df = df_raw[[ticker_a, ticker_b]].dropna()
    if len(pair_df) < 60: continue

    # Log-regression to find the hedge ratio (Beta)
    y, x = np.log(pair_df[ticker_a]), sm.add_constant(np.log(pair_df[ticker_b]))
    
    try:
        model = sm.OLS(y, x).fit()
        beta = model.params[ticker_b]
        
        # Calculate Spread and Z-Score
        spread = y - (beta * np.log(pair_df[ticker_b]) + model.params['const'])
        z_score_series = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
        curr_z = z_score_series.iloc[-1]
        pair_vol = spread.pct_change().std() * np.sqrt(252)

        with cols[i % 2]:
            st.subheader(f"📊 {ticker_a} vs {ticker_b}")
            strat_name, strat_desc, dte = get_instrument_type(pair_vol)

            # === DYNAMIC SIGNAL LOGIC (NO MORE 'A' OR 'B') ===
            if abs(curr_z) > ENTRY_Z:
                if curr_z < -ENTRY_Z:
                    # Ticker A is cheap relative to Ticker B
                    bg_color, border_color = "#d4edda", "#28a745"
                    signal_text = f"LONG SPREAD: BUY {ticker_a}"
                    instruction = f"🟢 BUY {ticker_a} / 🔴 SELL {ticker_b}"
                    hedge_on = ticker_a
                else:
                    # Ticker A is expensive relative to Ticker B
                    bg_color, border_color = "#f8d7da", "#dc3545"
                    signal_text = f"SHORT SPREAD: SELL {ticker_a}"
                    instruction = f"🔴 SELL {ticker_a} / 🟢 BUY {ticker_b}"
                    hedge_on = ticker_a

                st.markdown(f"""
                    <div style="background-color:{bg_color}; padding:20px; border-radius:10px; border: 3px solid {border_color}; color:black;">
                        <h2 style="margin:0; font-size:24px;">🔥 SIGNAL: {signal_text}</h2>
                        <hr style="border-top: 1px solid {border_color}; margin: 10px 0;">
                        <p style="font-size:20px; margin-bottom:5px;"><b>EXECUTION:</b> {instruction}</p>
                        <p style="font-size:16px; margin:0;"><b>RATIO:</b> 1 share of {ticker_a} per {round(beta, 2)} shares of {ticker_b}</p>
                        <p style="font-size:16px; margin-top:10px;"><b>HEDGE:</b> {strat_name} on <b>{hedge_on}</b> ({dte})</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info(f"⚪ **NO TRADE** for {ticker_a}/{ticker_b} | Z-Score: {curr_z:.2f}")

            # ==================== CHARTING ====================
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=z_score_series.index, y=z_score_series, name="Z-Score", line=dict(color='#00d1ff', width=2)))

            # Threshold Lines
            fig.add_hline(y=ENTRY_Z, line_dash="dash", line_color="red", annotation_text="SELL A")
            fig.add_hline(y=-ENTRY_Z, line_dash="dash", line_color="green", annotation_text="BUY A")
            fig.add_hline(y=0, line_dash="dot", line_color="white")

            fig.update_layout(
                template="plotly_dark", 
                height=350, 
                margin=dict(l=10, r=10, t=30, b=10),
                yaxis=dict(range=[-4, 4])
            )

            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"**Current Z:** {curr_z:.2f} | **Beta (Ratio):** {beta:.2f} | **Spread Vol:** {pair_vol:.1%}")
            st.divider()

    except Exception as e:
        with cols[i % 2]: st.error(f"Error on {ticker_a}/{ticker_b}: {str(e)}")

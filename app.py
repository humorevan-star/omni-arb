import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Omni-Arb Command Center", layout="wide", initial_sidebar_state="collapsed")

# ==========================================
# CUSTOM CSS FOR SLEEK UI
# ==========================================
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #fafafa; }
    .summary-card {
        background-color: #1e2129; padding: 15px; border-radius: 8px;
        border-left: 5px solid #00d1ff; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .signal-box-long {
        background-color: rgba(40, 167, 69, 0.1); border: 2px solid #28a745;
        padding: 20px; border-radius: 10px; margin-bottom: 15px;
    }
    .signal-box-short {
        background-color: rgba(220, 53, 69, 0.1); border: 2px solid #dc3545;
        padding: 20px; border-radius: 10px; margin-bottom: 15px;
    }
    h1, h2, h3, p { font-family: 'Inter', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 Omni-Arb Live Monitor v5.0 (Institutional)")
st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EDT | Engine: Rolling OLS & ADF Cointegration")
st.divider()

# ==========================================
# 1. SETUP & PARAMETERS
# ==========================================
PAIRS = [('XOM', 'CVX'), ('V', 'MA'), ('NVDA', 'AMD'), ('KO', 'PEP'), ('GOOGL', 'META')]
ENTRY_Z, EXIT_Z, STOP_Z = 2.25, 0.0, 3.5
ROLLING_WINDOW = 60

@st.cache_data(ttl=3600)
def get_data():
    tickers = list(set([t for p in PAIRS for t in p]))
    # yfinance returns multi-index for multiple tickers, we extract just the 'Close'
    data = yf.download(tickers, period="200d", interval="1d")['Close']
    return data

df_raw = get_data()

# ==========================================
# 2. DATA PROCESSING ENGINE (QUANT UPGRADED)
# ==========================================
processed_data = []
active_signals = []

for ticker_a, ticker_b in PAIRS:
    if ticker_a not in df_raw.columns or ticker_b not in df_raw.columns: 
        continue
    
    pair_df = df_raw[[ticker_a, ticker_b]].dropna()
    if len(pair_df) < ROLLING_WINDOW: 
        continue

    # 1. Rolling OLS for dynamic Beta (Prevents look-ahead bias)
    endog = np.log(pair_df[ticker_a])
    exog = sm.add_constant(np.log(pair_df[ticker_b]))
    
    try:
        rols = RollingOLS(endog, exog, window=ROLLING_WINDOW).fit()
        # Extract the rolling beta for ticker_b
        rolling_betas = rols.params[ticker_b]
        rolling_consts = rols.params['const']
        
        # Calculate dynamic spread
        spread = endog - (rolling_betas * np.log(pair_df[ticker_b]) + rolling_consts)
        spread = spread.dropna()
        
        # 2. Cointegration Check (ADF Test)
        # p-value < 0.05 means the spread is mean-reverting (safe to trade)
        adf_pval = adfuller(spread)[1]
        is_cointegrated = adf_pval < 0.05

        # 3. Z-Score Calculation
        z_score_series = (spread - spread.rolling(ROLLING_WINDOW).mean()) / spread.rolling(ROLLING_WINDOW).std()
        curr_z = z_score_series.iloc[-1]
        
        # Current Prices & Beta for execution sizing
        price_a = pair_df[ticker_a].iloc[-1]
        price_b = pair_df[ticker_b].iloc[-1]
        curr_beta = rolling_betas.iloc[-1]

        is_active = abs(curr_z) >= ENTRY_Z
        direction = "LONG" if curr_z <= -ENTRY_Z else "SHORT" if curr_z >= ENTRY_Z else "NEUTRAL"
        
        pair_info = {
            'pair': f"{ticker_a} / {ticker_b}",
            'a': ticker_a, 'b': ticker_b,
            'price_a': price_a, 'price_b': price_b,
            'curr_z': curr_z, 'beta': curr_beta, 
            'z_series': z_score_series,
            'is_active': is_active, 'direction': direction,
            'is_cointegrated': is_cointegrated, 'adf_pval': adf_pval
        }
        processed_data.append(pair_info)
        
        if is_active:
            active_signals.append(pair_info)
            
    except Exception as e:
        st.error(f"Error processing {ticker_a}/{ticker_b}: {str(e)}")

# ==========================================
# 3. TOP DASHBOARD: OPEN TRADES SUMMARY
# ==========================================
st.subheader("⚡ Active Trade Summary & Execution Points")

if not active_signals:
    st.info(f"No active signals. All Z-scores are currently between -{ENTRY_Z} and +{ENTRY_Z}. Waiting for deviations.")
else:
    # Use containers to wrap columns neatly if there are many active signals
    cols = st.columns(3) 
    for i, sig in enumerate(active_signals):
        with cols[i % 3]:
            color = "#28a745" if sig['direction'] == "LONG" else "#dc3545"
            bg_color = "rgba(40, 167, 69, 0.1)" if sig['direction'] == "LONG" else "rgba(220, 53, 69, 0.1)"
            
            # Plain English Z-Score Translation
            z_meaning = "Undervalued (Buy Spread)" if sig['direction'] == "LONG" else "Overvalued (Sell Spread)"
            coint_warning = "" if sig['is_cointegrated'] else "<p style='color:#ffc107; font-size: 12px;'>⚠️ WARNING: Pair not cointegrated recently.</p>"
            
            # Stock Execution Logic
            if sig['direction'] == "LONG":
                exec_text = f"<b>BUY</b> 100 shrs {sig['a']} @ ${sig['price_a']:.2f}<br><b>SELL</b> {int(100 * sig['beta'])} shrs {sig['b']} @ ${sig['price_b']:.2f}"
            else:
                exec_text = f"<b>SELL</b> 100 shrs {sig['a']} @ ${sig['price_a']:.2f}<br><b>BUY</b> {int(100 * sig['beta'])} shrs {sig['b']} @ ${sig['price_b']:.2f}"

            st.markdown(f"""
                <div style="background-color:{bg_color}; padding: 15px; border-radius: 8px; border-top: 4px solid {color}; margin-bottom: 15px;">
                    <h3 style="margin:0; font-size: 18px; color: {color};">{sig['direction']} {sig['a']} (vs {sig['b']})</h3>
                    
                    <div style="margin-top: 10px; padding-bottom: 5px; border-bottom: 1px solid rgba(255,255,255,0.1);">
                        <p style="margin:0; font-size: 14px; color:

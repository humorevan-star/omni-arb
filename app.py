# =============================================================================
# OMNI-ARB v6.5  |  Statistical Arbitrage Terminal (Rebalance Engine)
# =============================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
from datetime import datetime

# =============================================================================
# 0. CONFIG & THEME
# =============================================================================
st.set_page_config(page_title="Omni-Arb v6.5", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .main { background-color: #0b0e14; color: #e1e1e1; }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }
    .rebalance-box { 
        background: rgba(245, 166, 35, 0.1); 
        border: 1px solid rgba(245, 166, 35, 0.4); 
        padding: 10px; border-radius: 4px; margin-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# 1. PARAMETERS
# =============================================================================
PAIRS            = [('XOM', 'CVX'), ('V', 'MA'), ('NVDA', 'AMD'), ('KO', 'PEP'), ('GOOGL', 'META')]
ENTRY_Z          = 2.25
STOP_Z           = 3.5
EXIT_Z           = 0.0
ROLLING_WINDOW   = 60
STARTING_CAPITAL = 1000
REBALANCE_DAYS   = 2 # Medallion Hysteresis Frequency

# =============================================================================
# 2. SIZING & REBALANCE LOGIC
# =============================================================================
def compute_legs(sig, capital=STARTING_CAPITAL):
    price_a = float(sig["price_a"])
    price_b = float(sig["price_b"])
    beta    = max(abs(float(sig["beta"])), 0.01)

    # Step 1 — Risk-neutral dollar allocation
    dollar_a = capital / (1.0 + beta)
    dollar_b = capital - dollar_a

    # Step 2 — Ideal shares at 0.1 precision (CURRENT BETA)
    ideal_shares_a = max(0.1, round(dollar_a / price_a, 1))
    ideal_shares_b = max(0.1, round(dollar_b / price_b, 1))

    # Step 3 — Handle Rebalance Logic
    rebalance_msg = None
    ot = sig.get("open_trade")
    
    # Check if we are in an "Active Trade" state
    if ot and ot.get("entry_date"):
        days_in_trade = (datetime.now() - ot["entry_date"]).days
        
        # Every 2 days, check if current shares differ from ideal shares
        if days_in_trade > 0 and days_in_trade % REBALANCE_DAYS == 0:
            diff_b = round(ideal_shares_b - ot["entry_shares_b"], 1)
            if abs(diff_b) >= 0.1:
                action = "BUY" if diff_b > 0 else "SELL"
                rebalance_msg = f"REBALANCE: {action} {abs(diff_b)} shares of {sig['b']} to reset Beta neutrality."

    # Use entry-locked shares for P&L, but ideal shares for 'Current Sizing' display
    shares_a = ot["entry_shares_a"] if ot else ideal_shares_a
    shares_b = ot["entry_shares_b"] if ot else ideal_shares_b

    notional_a = round(shares_a * price_a, 2)
    notional_b = round(shares_b * price_b, 2)
    
    # Risk imbalance: val_a - (val_b / beta)
    risk_imbalance = round(notional_a - (notional_b / beta), 2)
    
    return {
        "shares_a": shares_a,
        "shares_b": shares_b,
        "notional_a": notional_a,
        "notional_b": notional_b,
        "total_cost": round(notional_a + notional_b, 2),
        "risk_imbalance": risk_imbalance,
        "rebalance_msg": rebalance_msg,
        "beta_at_entry": ot["entry_beta"] if ot else beta,
        "current_beta": beta
    }

# =============================================================================
# 3. DATA & PROCESSING (Injecting Beta at Entry)
# =============================================================================
@st.cache_data(ttl=3600)
def get_market_data():
    tickers = list(set([t for p in PAIRS for t in p]))
    return yf.download(tickers, period="750d")["Close"]

def process_pairs(df_raw):
    processed, active = [], []
    for ticker_a, ticker_b in PAIRS:
        pair_df = df_raw[[ticker_a, ticker_b]].dropna()
        y, x = np.log(pair_df[ticker_a]), sm.add_constant(np.log(pair_df[ticker_b]))
        model = RollingOLS(y, x, window=ROLLING_WINDOW).fit()
        betas, z_series = model.params[ticker_b], pd.Series() # Simplified for brevity

        # (Insert your Z-score calculation here as per v6.0)
        # ... [Z-Score Calculation Logic] ...
        
        # State Machine Logic (LOCKED ENTRY BETA)
        # When entering a trade, we must store the Beta at that moment:
        # open_trade = { ..., "entry_beta": betas.loc[entry_date], ... }
        
        # (Remaining processing logic from v6.0)
    return processed, active

# =============================================================================
# 4. RENDER UI (Adding Rebalance Alert)
# =============================================================================
def render_trade_card(sig):
    legs = compute_legs(sig)
    
    # Add Rebalance Alert Box if message exists
    if legs["rebalance_msg"]:
        st.markdown(f"""
            <div class="rebalance-box">
                <span style="color:#f5a623; font-weight:bold; font-family:monospace;">⏳ 2-DAY REBALANCE ALERT</span><br>
                <span style="color:#e1e1e1; font-size:12px;">{legs['rebalance_msg']}</span>
            </div>
        """, unsafe_allow_html=True)

    # (Insert your existing Card Rendering logic here)

# =============================================================================
# MAIN EXECUTION
# =============================================================================
st.title("📟 OMNI-ARB v6.5 | StatArb Terminal")
data = get_market_data()
processed, active = process_pairs(data)

for p in active:
    render_trade_card(p)
    st.plotly_chart(render_pair_chart(p), use_container_width=True)

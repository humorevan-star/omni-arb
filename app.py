import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import plotly.graph_objects as go
from datetime import datetime

# =============================================================================
# 0. GLOBAL SETTINGS & UI THEME
# =============================================================================
st.set_page_config(page_title="Omni-Arb v7.5", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
    .main { background-color: #0b0e14; color: #e1e1e1; }
    .stMetric { background: rgba(255,255,255,0.03); padding: 15px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); }
    code { font-family: 'IBM Plex Mono', monospace !important; color: #00ffcc !important; }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; }
    </style>
    """, unsafe_allow_html=True)

# Strategy Hyper-parameters
PORTFOLIO_TOTAL = 1000.0
EQUITY_ALLOC    = 0.60  # $600 base
OPTION_ALLOC    = 0.40  # $400 leverage
ENTRY_Z         = 2.25
EXIT_Z          = 0.25
STOP_Z          = 3.75
STRIKE_OFFSET   = 0.025 # 2.5% OTM
LOOKBACK_DAYS   = 730   # 2 Years

# Universe
PAIRS = [('XOM', 'CVX'), ('V', 'MA'), ('NVDA', 'AMD'), ('KO', 'PEP'), ('MSTR', 'BTC-USD')]

# =============================================================================
# 1. DATA & MATH ENGINE
# =============================================================================
@st.cache_data(ttl=3600)
def get_market_data():
    tickers = list(set([t for p in PAIRS for t in p]))
    df = yf.download(tickers, period="3y", interval="1d")["Close"]
    return df.ffill().dropna()

def calculate_metrics(df, t1, t2):
    """Core Stat-Arb logic with Rolling OLS."""
    y = np.log(df[t1])
    x = sm.add_constant(np.log(df[t2]))
    model = RollingOLS(y, x, window=60).fit()
    
    beta = model.params[t2]
    const = model.params["const"]
    spread = y - (beta * np.log(df[t2]) + const)
    z_score = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
    
    return z_score, beta, spread

# =============================================================================
# 2. THE HYBRID BACKTESTER
# =============================================================================
def run_backtest(df):
    ledger = []
    
    for t1, t2 in PAIRS:
        if t1 not in df.columns or t2 not in df.columns: continue
        
        z, beta_series, _ = calculate_metrics(df, t1, t2)
        in_pos = False
        entry_idx = None
        direction = None
        
        for i in range(60, len(z)):
            curr_z = z.iloc[i]
            dt = z.index[i]
            
            if not in_pos:
                if curr_z >= ENTRY_Z:
                    in_pos, direction, entry_idx = True, "SHORT", i
                elif curr_z <= -ENTRY_Z:
                    in_pos, direction, entry_idx = True, "LONG", i
            else:
                # Exit Logic: Reversion, Stop, or Time-out (21 days)
                days_held = i - entry_idx
                if (direction == "LONG" and curr_z >= -EXIT_Z) or \
                   (direction == "SHORT" and curr_z <= EXIT_Z) or \
                   abs(curr_z) >= STOP_Z or days_held >= 21:
                    
                    # P&L Calculation
                    p_a_entry, p_a_exit = df[t1].iloc[entry_idx], df[t1].iloc[i]
                    p_b_entry, p_b_exit = df[t2].iloc[entry_idx], df[t2].iloc[i]
                    beta = beta_series.iloc[entry_idx]
                    
                    ret_a = (p_a_exit / p_a_entry) - 1
                    ret_b = (p_b_exit / p_b_entry) - 1
                    
                    # Spread return (Beta-neutral)
                    spread_ret = (ret_a - beta * ret_b) if direction == "LONG" else -(ret_a - beta * ret_b)
                    
                    # 60% Stock Component
                    stock_pnl = (PORTFOLIO_TOTAL * EQUITY_ALLOC) * spread_ret
                    
                    # 40% Vertical Component (Simulated 4x Leverage on the spread move)
                    option_pnl = (PORTFOLIO_TOTAL * OPTION_ALLOC) * (spread_ret * 4.0)
                    
                    ledger.append({
                        "Date": dt,
                        "Pair": f"{t1}/{t2}",
                        "Type": direction,
                        "Stock_PnL": stock_pnl,
                        "Option_PnL": option_pnl,
                        "Total_PnL": stock_pnl + option_pnl
                    })
                    in_pos = False
                    
    if not ledger: return pd.DataFrame()
    
    # CRITICAL: Sort by date to fix the "messed up" chart logic
    report = pd.DataFrame(ledger).sort_values("Date")
    report["Cum_PnL"] = report["Total_PnL"].cumsum()
    report["Stock_Only_Cum"] = report["Stock_PnL"].cumsum()
    return report

# =============================================================================
# 3. MAIN INTERFACE
# =============================================================================
def main():
    st.title("Omni-Arb v7.5 | Institutional Terminal")
    st.caption(f"Strategy: 60/40 Hybrid Allocation | Z-Entry: {ENTRY_Z} | Vertical Offset: {STRIKE_OFFSET*100}%")

    df = get_market_data()
    
    # --- ACTIVE SIGNALS SECTION ---
    st.subheader("Live Execution Signals")
    cols = st.columns(3)
    active_count = 0
    
    for i, (t1, t2) in enumerate(PAIRS):
        z, beta, _ = calculate_metrics(df, t1, t2)
        curr_z = z.iloc[-1]
        
        if abs(curr_z) >= 1.5: # Show pairs nearing or at trigger
            active_count += 1
            with cols[active_count % 3]:
                status = "🚨 TRIGGER" if abs(curr_z) >= ENTRY_Z else "⏳ MONITOR"
                st.markdown(f"**{t1}/{t2}** | `{status}`")
                st.metric("Z-Score", f"{curr_z:.2f}", delta=f"{curr_z - z.iloc[-2]:.2f}")
                
                if abs(curr_z) >= ENTRY_Z:
                    direction = "LONG SPREAD" if curr_z < 0 else "SHORT SPREAD"
                    st.info(f"**Action:** {direction}\n\n**Verticals:** {t1} @ ${df[t1].iloc[-1]*(1+STRIKE_OFFSET if curr_z < 0 else 1-STRIKE_OFFSET):.2f}")

    if active_count == 0:
        st.write("All pairs currently within neutral bands.")

    # --- WATCHLIST CHARTS ---
    st.divider()
    st.subheader("Pair Watchlist (2-Year Z-Scores)")
    w_cols = st.columns(2)
    for i, (t1, t2) in enumerate(PAIRS):
        z, _, _ = calculate_metrics(df, t1, t2)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=z.index, y=z, name="Z-Score", line=dict(color="#00d1ff")))
        fig.add_hline(y=ENTRY_Z, line_dash="dash", line_color="#ff4b4b")
        fig.add_hline(y=-ENTRY_Z, line_dash="dash", line_color="#00ffcc")
        fig.update_layout(template="plotly_dark", height=250, margin=dict(l=0,r=0,t=0,b=0), showlegend=False)
        w_cols[i % 2].plotly_chart(fig, use_container_width=True)

    # --- BACKTEST SECTION (MOVED TO BOTTOM) ---
    st.divider()
    st.header("Strategic Performance Backtest")
    report = run_backtest(df)
    
    if not report.empty:
        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Profit", f"${report['Total_PnL'].sum():,.2f}")
        m2.metric("Win Rate", f"{(report['Total_PnL'] > 0).mean():.1%}")
        m3.metric("Option Contribution", f"${report['Option_PnL'].sum():,.2f}")
        m4.metric("Avg Trade P&L", f"${report['Total_PnL'].mean():.2f}")
        
        # Unified Equity Chart
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=report["Date"], y=report["Cum_PnL"], 
                                   name="Hybrid Portfolio ($1k)", line=dict(color="#00ffcc", width=3),
                                   fill='tozeroy'))
        fig_bt.add_trace(go.Scatter(x=report["Date"], y=report["Stock_Only_Cum"], 
                                   name="Stock Only ($600)", line=dict(color="#888", dash="dot")))
        
        fig_bt.update_layout(template="plotly_dark", height=450, 
                            title="Consolidated Equity Curve (Past 2 Years)",
                            yaxis_title="Cumulative P&L ($)",
                            xaxis_title="Timeline")
        st.plotly_chart(fig_bt, use_container_width=True)
        
        # Trade History Table
        with st.expander("View Full Trade Ledger"):
            st.dataframe(report.sort_values("Date", ascending=False), use_container_width=True)
    else:
        st.warning("Insufficient data to generate backtest. Check ticker availability.")

if __name__ == "__main__":
    main()

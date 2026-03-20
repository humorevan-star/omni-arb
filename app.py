import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import plotly.graph_objects as go
from datetime import datetime

# =============================================================================
# 0. DASHBOARD CONFIG
# =============================================================================
st.set_page_config(page_title="Omni-Arb v8.0", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap');
    .main { background-color: #0b0e14; color: #e1e1e1; }
    .stMetric { background: rgba(255,255,255,0.03); padding: 15px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); }
    code { font-family: 'IBM Plex Mono', monospace !important; color: #00ffcc !important; }
    .status-card { background: rgba(0,255,204,0.05); border-left: 5px solid #00ffcc; padding: 15px; border-radius: 4px; margin-bottom: 10px; }
    </style>
    """, unsafe_allow_html=True)

# Strategy Parameters
PORTFOLIO_TOTAL = 1000.0
EQUITY_ALLOC    = 0.60  # $600 
OPTION_ALLOC    = 0.40  # $400 
ENTRY_Z         = 2.25
EXIT_Z          = 0.20
STOP_Z          = 3.75
STRIKE_OFFSET   = 0.025 # 2.5% OTM for Verticals

PAIRS = [('XOM', 'CVX'), ('V', 'MA'), ('NVDA', 'AMD'), ('KO', 'PEP'), ('MSTR', 'BTC-USD')]

# =============================================================================
# 1. CORE DATA ENGINE
# =============================================================================
@st.cache_data(ttl=3600)
def get_market_data():
    tickers = list(set([t for p in PAIRS for t in p]))
    df = yf.download(tickers, period="3y", interval="1d")["Close"]
    return df.ffill().dropna()

def calculate_pair_stats(df, t1, t2):
    """Calculates Rolling Beta and Z-Score for the pair."""
    y = np.log(df[t1])
    x = sm.add_constant(np.log(df[t2]))
    model = RollingOLS(y, x, window=60).fit()
    
    beta = model.params[t2]
    const = model.params["const"]
    spread = y - (beta * np.log(df[t2]) + const)
    z_score = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
    
    return z_score, beta

# =============================================================================
# 2. HYBRID PERFORMANCE BACKTESTER
# =============================================================================
def run_hybrid_backtest(df):
    ledger = []
    
    for t1, t2 in PAIRS:
        if t1 not in df.columns or t2 not in df.columns: continue
        z, beta_series = calculate_pair_stats(df, t1, t2)
        in_pos = False
        entry_idx = None
        direction = None
        
        for i in range(60, len(z)):
            curr_z = z.iloc[i]
            if not in_pos:
                if curr_z >= ENTRY_Z: in_pos, direction, entry_idx = True, "SHORT", i
                elif curr_z <= -ENTRY_Z: in_pos, direction, entry_idx = True, "LONG", i
            else:
                days_held = i - entry_idx
                if (direction == "LONG" and curr_z >= -EXIT_Z) or \
                   (direction == "SHORT" and curr_z <= EXIT_Z) or \
                   abs(curr_z) >= STOP_Z or days_held >= 21:
                    
                    # P&L Logic
                    ret_a = (df[t1].iloc[i] / df[t1].iloc[entry_idx]) - 1
                    ret_b = (df[t2].iloc[i] / df[t2].iloc[entry_idx]) - 1
                    beta = beta_series.iloc[entry_idx]
                    
                    spread_ret = (ret_a - beta * ret_b) if direction == "LONG" else -(ret_a - beta * ret_b)
                    
                    # 60/40 Hybrid Split
                    stock_pnl = (PORTFOLIO_TOTAL * EQUITY_ALLOC) * spread_ret
                    # Verticals leverage: modeling a 4.5x multiplier on the spread return
                    option_pnl = (PORTFOLIO_TOTAL * OPTION_ALLOC) * (spread_ret * 4.5)
                    
                    ledger.append({
                        "Date": z.index[i],
                        "Pair": f"{t1}/{t2}",
                        "Type": direction,
                        "Stock_PnL": stock_pnl,
                        "Option_PnL": option_pnl,
                        "Total_PnL": stock_pnl + option_pnl
                    })
                    in_pos = False
                    
    if not ledger: return pd.DataFrame()
    
    report = pd.DataFrame(ledger).sort_values("Date")
    report["Cum_PnL"] = report["Total_PnL"].cumsum()
    report["Stock_Only_Cum"] = report["Stock_PnL"].cumsum()
    return report

# =============================================================================
# 3. DASHBOARD RENDERING
# =============================================================================
def main():
    st.title("Omni-Arb Terminal v8.0")
    st.caption(f"Status: Active | Strategy: Hybrid 60/40 Stocks & Verticals | Updated: {datetime.now().strftime('%H:%M:%S')}")

    df = get_market_data()

    # --- SECTION A: LIVE EXECUTION DASHBOARD ---
    st.subheader("Front-End Signals")
    sig_cols = st.columns(len(PAIRS))
    
    for i, (t1, t2) in enumerate(PAIRS):
        z, beta = calculate_pair_stats(df, t1, t2)
        curr_z = z.iloc[-1]
        
        with sig_cols[i]:
            # Color-coded trigger logic
            color = "white"
            if curr_z <= -ENTRY_Z: color = "#00ffcc" # Neon Green
            elif curr_z >= ENTRY_Z: color = "#ff4b4b" # Red
            
            st.markdown(f"**{t1}/{t2}**")
            st.metric("Z-Score", f"{curr_z:.2f}", delta=f"{curr_z - z.iloc[-2]:.2f}", delta_color="normal")
            
            if abs(curr_z) >= ENTRY_Z:
                direction = "LONG SPREAD" if curr_z < 0 else "SHORT SPREAD"
                st.markdown(f"""<div class="status-card" style="border-color:{color}">
                    <b>ACTION: {direction}</b><br>
                    <small>Buy {t1} @ {df[t1].iloc[-1]:.1f}<br>
                    Sell {t2} @ {df[t2].iloc[-1]:.1f}</small>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("<small style='color:#4a5568'>Status: Monitoring</small>", unsafe_allow_html=True)

    # --- SECTION B: THE WATCHLIST (CHART GRID) ---
    st.divider()
    st.subheader("Global Watchlist")
    w_cols = st.columns(2)
    for i, (t1, t2) in enumerate(PAIRS):
        z, _ = calculate_pair_stats(df, t1, t2)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=z.index, y=z, name="Z-Score", line=dict(color="#00d1ff", width=1.5)))
        fig.add_hline(y=ENTRY_Z, line_dash="dash", line_color="#ff4b4b", opacity=0.5)
        fig.add_hline(y=-ENTRY_Z, line_dash="dash", line_color="#00ffcc", opacity=0.5)
        fig.update_layout(template="plotly_dark", height=240, margin=dict(l=10, r=10, t=20, b=10), 
                          title=f"{t1}/{t2} Variance", title_font_size=12)
        w_cols[i % 2].plotly_chart(fig, use_container_width=True)

    # --- SECTION C: HYBRID BACKTEST (AT BOTTOM) ---
    st.divider()
    st.header("Hybrid Strategy Performance")
    report = run_hybrid_backtest(df)
    
    if not report.empty:
        # P&L Stats
        total_pnl = report['Total_PnL'].sum()
        win_rate = (report['Total_PnL'] > 0).mean()
        
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Net Portfolio Return", f"${total_pnl:,.2f}", f"{(total_pnl/1000)*100:.1f}%")
        m2.metric("Win Rate", f"{win_rate:.1%}")
        m3.metric("Option P&L Contribution", f"${report['Option_PnL'].sum():,.2f}")
        m4.metric("Average Hybrid Trade", f"${report['Total_PnL'].mean():.2f}")

        # Chronological Equity Chart
        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=report["Date"], y=report["Cum_PnL"], 
                                   name="Hybrid (60% Stock / 40% Vertical)", 
                                   line=dict(color="#00ffcc", width=3), fill='tozeroy'))
        fig_bt.add_trace(go.Scatter(x=report["Date"], y=report["Stock_Only_Cum"], 
                                   name="Stock Only Benchmark", 
                                   line=dict(color="#8892a4", dash="dot")))
        
        fig_bt.update_layout(template="plotly_dark", height=450, 
                            title="2-Year Portfolio Equity Curve",
                            yaxis_title="Total P&L ($)", xaxis_title="Timeline")
        st.plotly_chart(fig_bt, use_container_width=True)
        
        with st.expander("Detailed Trade Ledger"):
            st.dataframe(report.sort_values("Date", ascending=False), use_container_width=True)
    else:
        st.error("Error: Data engine failed to calculate backtest. Verify tickers.")

if __name__ == "__main__":
    main()

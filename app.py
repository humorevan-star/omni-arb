import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
import plotly.graph_objects as go
from datetime import datetime, timedelta

# =============================================================================
# CONFIG & HYPER-PARAMETERS
# =============================================================================
st.set_page_config(page_title="Omni-Arb v7.0", layout="wide")

PORTFOLIO_TOTAL = 1000.0
EQUITY_ALLOC    = 0.60  # $600
OPTION_ALLOC    = 0.40  # $400
ENTRY_Z         = 2.25
EXIT_Z          = 0.25
STOP_Z          = 3.75
STRIKE_OFFSET   = 0.025 # 2.5% OTM Fine-tuning
LOOKBACK        = 730   # 2 Years

PAIRS = [('XOM', 'CVX'), ('V', 'MA'), ('NVDA', 'AMD'), ('KO', 'PEP')]

# =============================================================================
# DATA ENGINE
# =============================================================================
@st.cache_data(ttl=3600)
def get_historical_data():
    tickers = list(set([t for p in PAIRS for t in p]))
    df = yf.download(tickers, period="3y", interval="1d")["Close"]
    return df.dropna()

def get_pair_stats(df, t1, t2):
    y = np.log(df[t1])
    x = sm.add_constant(np.log(df[t2]))
    model = RollingOLS(y, x, window=60).fit()
    
    betas = model.params[t2]
    consts = model.params["const"]
    spread = y - (betas * np.log(df[t2]) + consts)
    z_score = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
    
    return z_score, betas, spread

# =============================================================================
# HYBRID BACKTESTER (STOCKS + VERTICALS)
# =============================================================================
def run_hybrid_backtest(df):
    all_trades = []
    
    for t1, t2 in PAIRS:
        z, betas, spread = get_pair_stats(df, t1, t2)
        in_pos = False
        entry_idx = None
        direction = None
        
        for i in range(len(z)):
            curr_z = z.iloc[i]
            curr_dt = z.index[i]
            
            if not in_pos:
                if curr_z >= ENTRY_Z:
                    in_pos, direction, entry_idx = True, "SHORT", i
                elif curr_z <= -ENTRY_Z:
                    in_pos, direction, entry_idx = True, "LONG", i
            else:
                # Exit Logic: Reversion to mean or Stop Loss
                days_held = i - entry_idx
                if (direction == "LONG" and curr_z >= -EXIT_Z) or \
                   (direction == "SHORT" and curr_z <= EXIT_Z) or \
                   abs(curr_z) >= STOP_Z or days_held > 30:
                    
                    # --- Stock P&L Calculation ---
                    # Using log-spread delta as a proxy for $600 allocation
                    ret_a = np.log(df[t1].iloc[i] / df[t1].iloc[entry_idx])
                    ret_b = np.log(df[t2].iloc[i] / df[t2].iloc[entry_idx])
                    beta_entry = betas.iloc[entry_idx]
                    
                    # Net spread return
                    spread_ret = (ret_a - beta_entry * ret_b) if direction == "LONG" else -(ret_a - beta_entry * ret_b)
                    stock_pnl = (EQUITY_ALLOC * PORTFOLIO_TOTAL) * spread_ret
                    
                    # --- Vertical Option P&L Calculation ---
                    # Simulating a 2.5% OTM Vertical Spread with 3.0x leverage on the spread move
                    # Verticals decay, but gamma/delta capture is high during reversion
                    option_pnl = (OPTION_ALLOC * PORTFOLIO_TOTAL) * (spread_ret * 3.5) # Leverage factor
                    
                    all_trades.append({
                        "Pair": f"{t1}/{t2}",
                        "Entry": z.index[entry_idx].date(),
                        "Exit": curr_dt.date(),
                        "Type": direction,
                        "Stock P&L": round(stock_pnl, 2),
                        "Option P&L": round(option_pnl, 2),
                        "Total P&L": round(stock_pnl + option_pnl, 2)
                    })
                    in_pos = False
                    
    return pd.DataFrame(all_trades)

# =============================================================================
# UI LAYOUT
# =============================================================================
def main():
    st.title("Omni-Arb v7.0 | Institutional Hybrid Terminal")
    
    df = get_historical_data()
    trades_df = run_hybrid_backtest(df)
    
    # --- Top Level Metrics ---
    total_net = trades_df["Total P&L"].sum()
    win_rate = (trades_df["Total P&L"] > 0).mean()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net Cumulative P&L", f"${total_net:,.2f}", f"{total_net/10:.1f}%")
    c2.metric("Win Rate", f"{win_rate:.1%}")
    c3.metric("Total Trades", len(trades_df))
    c4.metric("Avg Trade", f"${trades_df['Total P&L'].mean():.2f}")

    # --- Active Signal Check ---
    st.subheader("Live Pair Analysis")
    for t1, t2 in PAIRS:
        z, beta, _ = get_pair_stats(df, t1, t2)
        curr_z = z.iloc[-1]
        
        if abs(curr_z) > 1.5: # Show high-interest pairs
            color = "green" if curr_z < -ENTRY_Z else "red" if curr_z > ENTRY_Z else "white"
            with st.expander(f"{t1}/{t2} - Current Z: {curr_z:.2f}", expanded=(abs(curr_z) > ENTRY_Z)):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.write(f"**Stock Leg ($600)**")
                    st.write(f"Pair Direction: {'LONG SPREAD' if curr_z < 0 else 'SHORT SPREAD'}")
                    st.write(f"Beta: `{beta.iloc[-1]:.2f}`")
                with col_b:
                    st.write(f"**Vertical Leg ($400)**")
                    strike_a = df[t1].iloc[-1] * (1 + STRIKE_OFFSET if curr_z < 0 else 1 - STRIKE_OFFSET)
                    st.write(f"Target Strike {t1}: `${strike_a:.2f}`")
                    st.write(f"Spread Style: Vertical Bull/Bear")

    # --- Backtest Charts ---
    st.divider()
    st.subheader("2-Year Strategy Backtest (Stock vs. Options)")
    
    trades_df["Cum P&L"] = trades_df["Total P&L"].cumsum()
    trades_df["Stock Only"] = trades_df["Stock P&L"].cumsum()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trades_df["Exit"], y=trades_df["Cum P&L"], name="Hybrid (60/40)", line=dict(color="#00ffcc", width=3)))
    fig.add_trace(go.Scatter(x=trades_df["Exit"], y=trades_df["Stock Only"], name="Stocks Only", line=dict(color="#888", dash="dot")))
    
    fig.update_layout(template="plotly_dark", height=450, xaxis_title="Trade Exit Date", yaxis_title="Profit ($)")
    st.plotly_chart(fig, use_container_width=True)

    # --- Trade Ledger ---
    st.subheader("Historical Trade Ledger")
    st.dataframe(trades_df.sort_values("Exit", ascending=False), use_container_width=True)

if __name__ == "__main__":
    main()

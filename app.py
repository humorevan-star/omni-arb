 import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go

st.set_page_config(page_title="Omni-Arb Dashboard", layout="wide")
st.title("🚀 Omni-Arb Live Monitor v2.9 – PROFESSIONAL CHARTS EDITION")

st.sidebar.header("📌 How to Use This Dashboard")
st.sidebar.markdown("""
- **Z-Score** = how far the pair has diverged.
- **ENTRY** → |Z| > 2.25  
- **EXIT / COVER** → Z returns to 0  
- **STOP LOSS** → |Z| > 3.5  
- Every signal shows **ALL 3 strategies** (Recommended + Vertical + Full Stock Pair)
- **Charts fully cleaned**: No overlapping text anywhere. Annotations split (right for positive/exit, left for negative). Extra spacing, taller plots, clean fonts.
""")

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
    if len(pair_df) < 60:
        with cols[i % 2]:
            st.warning(f"⛔ {a}-{b}: Not enough data")
        continue

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

            # === SIGNAL BLOCK ===
            if abs(curr_z) > ENTRY_Z:
                direction = "SHORT THE SPREAD" if curr_z > 0 else "LONG THE SPREAD"
                color = "red" if curr_z > 0 else "green"
                st.markdown(f"### :{color}[{direction} – ENTER NOW]")

                st.success(f"**RECOMMENDED STRATEGY:** {strat_name} ({dte})")

                main_ticker = a

                # PRIMARY TRADE
                if strat_name == "RATIO BACKSPREAD":
                    opt_type = "Puts" if curr_z > 0 else "Calls"
                    st.info(f"""
**TRADE OPTIONS ON {main_ticker}**  
**RATIO BACKSPREAD (60-70 Days Out)**

1. **Sell (1)** {main_ticker} At-the-Money {opt_type}  
2. **Buy (2)** {main_ticker} Out-of-the-Money {opt_type}  

**Expiration:** 60-70 Days Out  
**Ratio:** 1:2  
**Note:** Unlimited profit if big move in expected direction | Limited risk opposite way  
**Does NOT include the other leg ({b})**
""")

                elif strat_name == "VERTICAL SPREAD":
                    if curr_z < 0:
                        st.info(f"""
**TRADE OPTIONS ON {main_ticker}**  
**VERTICAL SPREAD (30-45 Days Out)**

**Bull Call Debit Spread**  
1. **Buy 1** {main_ticker} At-the-Money Call  
2. **Sell 1** {main_ticker} Out-of-the-Money Call (higher strike)

**Expiration:** 30-45 Days Out  
**Max Loss:** Net debit | **Max Profit:** Limited
""")
                    else:
                        st.info(f"""
**TRADE OPTIONS ON {main_ticker}**  
**VERTICAL SPREAD (30-45 Days Out)**

**Bear Put Debit Spread**  
1. **Buy 1** {main_ticker} At-the-Money Put  
2. **Sell 1** {main_ticker} Out-of-the-Money Put (lower strike)

**Expiration:** 30-45 Days Out  
**Max Loss:** Net debit | **Max Profit:** Limited
""")
                else:
                    action_a = "Short" if curr_z > 0 else "Buy"
                    action_b = "Buy" if curr_z > 0 else "Short"
                    st.info(f"""
**STOCK-ONLY TRADE (Both Legs)**

1. **{action_a} 100 shares of {a}**  
2. **{action_b} 100 shares of {b}**

**No expiration** – Hold until Z returns to 0 or stop loss
""")

                st.caption(f"**ENTRY** |Z| > {ENTRY_Z}  **EXIT** Z = 0  **STOP** |Z| > {STOP_Z}")

                # 2 ALTERNATIVE PLAYS
                with st.expander("🔄 2 Alternative Plays (Vertical + Full Stock Pair)", expanded=True):
                    st.markdown("**These are the other 2 best options** for the same signal.")

                    st.subheader("Alternative 1: Vertical Spread")
                    if curr_z < 0:
                        st.info(f"""
**VERTICAL SPREAD on {main_ticker} (30-45 Days Out)**

**Bull Call Debit Spread**  
1. **Buy 1** {main_ticker} At-the-Money Call  
2. **Sell 1** {main_ticker} Out-of-the-Money Call (higher strike)

**Expiration:** 30-45 Days Out  
**Max Loss:** Net debit | **Max Profit:** Limited
""")
                    else:
                        st.info(f"""
**VERTICAL SPREAD on {main_ticker} (30-45 Days Out)**

**Bear Put Debit Spread**  
1. **Buy 1** {main_ticker} At-the-Money Put  
2. **Sell 1** {main_ticker} Out-of-the-Money Put (lower strike)

**Expiration:** 30-45 Days Out  
**Max Loss:** Net debit | **Max Profit:** Limited
""")

                    st.subheader("Alternative 2: Pure Stock Trade (Full Pair)")
                    action_a = "Short" if curr_z > 0 else "Buy"
                    action_b = "Buy" if curr_z > 0 else "Short"
                    st.info(f"""
**STOCK-ONLY (Both Legs of the Pair)**

1. **{action_a} 100 shares of {a}**  
2. **{action_b} 100 shares of {b}**

**No expiration** – Hold until Z returns to 0 or stop loss  
**Includes the other leg ({b})**
""")

            else:
                st.info(f"⚪ **NO TRADE** – Z-Score: {curr_z:.2f}")
                st.caption(f"Strategy type: {strat_name} | Vol: {pair_vol:.1%}")

            # ==================== CLEAN PROFESSIONAL CHART – ZERO OVERLAP ====================
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=z_score_series.index,
                y=z_score_series,
                name="Z-Score",
                line=dict(color='cyan', width=2)
            ))

            # Clean horizontal lines (no built-in annotations)
            fig.add_hline(y=ENTRY_Z, line_dash="dash", line_color="red")
            fig.add_hline(y=-ENTRY_Z, line_dash="dash", line_color="red")
            fig.add_hline(y=0, line_dash="dot", line_color="lime")
            fig.add_hline(y=STOP_Z, line_dash="dash", line_color="orange")
            fig.add_hline(y=-STOP_Z, line_dash="dash", line_color="orange")

            # Current Z marker
            fig.add_trace(go.Scatter(
                x=[z_score_series.index[-1]],
                y=[curr_z],
                mode="markers+text",
                marker=dict(size=16, color="yellow", symbol="star"),
                text=[f"NOW<br>{curr_z:.2f}"],
                textposition="top right",
                name="Current Z",
                textfont=dict(size=12, color="white")
            ))

            # Professional annotations – positive/exit on RIGHT, negative on LEFT (no overlap ever)
            fig.update_layout(
                template="plotly_dark",
                height=340,
                margin=dict(l=40, r=150, t=40, b=40),   # extra right margin for clean text
                title="Z-Score History – Clear Entry / Exit / Stop Signals",
                yaxis_title="Z-Score",
                showlegend=True,
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                annotations=[
                    # RIGHT SIDE
                    dict(xref="paper", x=1.03, y=ENTRY_Z, text="ENTRY SHORT SPREAD", showarrow=False,
                         font=dict(color="red", size=11), xanchor="left", yanchor="middle"),
                    dict(xref="paper", x=1.03, y=STOP_Z, text="STOP LOSS", showarrow=False,
                         font=dict(color="orange", size=11), xanchor="left", yanchor="middle"),
                    dict(xref="paper", x=1.03, y=0, text="EXIT / COVER POSITION", showarrow=False,
                         font=dict(color="lime", size=11), xanchor="left", yanchor="middle"),
                    # LEFT SIDE
                    dict(xref="paper", x=-0.02, y=-ENTRY_Z, text="ENTRY LONG SPREAD", showarrow=False,
                         font=dict(color="red", size=11), xanchor="right", yanchor="middle"),
                    dict(xref="paper", x=-0.02, y=-STOP_Z, text="STOP LOSS", showarrow=False,
                         font=dict(color="orange", size=11), xanchor="right", yanchor="middle"),
                ]
            )

            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"**Current Z:** {curr_z:.2f} | **Spread Vol:** {pair_vol:.1%} | 60-day rolling")
            st.divider()

    except Exception as e:
        with cols[i % 2]:
            st.error(f"Error: {str(e)}")

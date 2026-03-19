import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go

st.set_page_config(page_title="Omni-Arb Dashboard", layout="wide")
st.title("🚀 Omni-Arb Live Monitor v2.6 – FOOLPROOF EDITION")

st.sidebar.header("📌 How to Use This Dashboard")
st.sidebar.markdown("""
- **Z-Score** = how far the pair has diverged (log-spread standardized).
- **ENTRY** → |Z| > 2.25  
- **EXIT / COVER** → Z returns to 0  
- **STOP LOSS** → |Z| > 3.5 (exit immediately, take the loss)
- All trades are **exactly** described below with ticker, legs, ratio, and exact DTE.
- Chart now has **clear visual signals** (colored lines + labels).
""")

# 1. SETUP – UNCHANGED THRESHOLDS
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
            st.warning(f"⛔ {a}-{b}: Not enough data ({len(pair_df)} days)")
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

                st.success(f"**STRATEGY:** {strat_name} ({dte})")

                # ==================== EXACT TRADE INSTRUCTIONS ====================
                main_ticker = a

                if strat_name == "RATIO BACKSPREAD":
                    opt_type = "Puts" if curr_z > 0 else "Calls"   # Bearish = Puts, Bullish = Calls
                    st.info(f"""
**TRADE OPTIONS ON {main_ticker}**  
**RATIO BACKSPREAD (60-70 Days Out)**

1. **Sell (1)** At-the-Money {opt_type}  
2. **Buy (2)** Out-of-the-Money {opt_type}  

**Expiration:** 60-70 Days Out  
**Ratio:** 1:2  
**Note:** Unlimited profit if big move in expected direction | Limited risk opposite way
""")

                elif strat_name == "VERTICAL SPREAD":
                    if curr_z < 0:  # LONG SPREAD → Bullish on A
                        st.info(f"""
**TRADE OPTIONS ON {main_ticker}**  
**VERTICAL SPREAD (30-45 Days Out)**

**Bull Call Debit Spread**  
1. Buy 1 At-the-Money Call  
2. Sell 1 Out-of-the-Money Call (higher strike)

**Expiration:** 30-45 Days Out  
**Max Loss:** Net debit | **Max Profit:** Limited
""")
                    else:  # SHORT SPREAD → Bearish on A
                        st.info(f"""
**TRADE OPTIONS ON {main_ticker}**  
**VERTICAL SPREAD (30-45 Days Out)**

**Bear Put Debit Spread**  
1. Buy 1 At-the-Money Put  
2. Sell 1 Out-of-the-Money Put (lower strike)

**Expiration:** 30-45 Days Out  
**Max Loss:** Net debit | **Max Profit:** Limited
""")

                else:  # PURE STOCKS
                    st.info(f"""
**STOCK-ONLY TRADE (No Options)**

**{a}** : {'Short' if curr_z > 0 else 'Buy'}  
**{b}** : {'Buy' if curr_z > 0 else 'Short'}

**No expiration** – Hold until Z returns to 0 or stop loss
""")

                # General rules (always shown when in trade)
                st.caption(f"""
**ENTRY** |Z| > {ENTRY_Z}  **EXIT/COVER** Z = 0  **STOP LOSS** |Z| > {STOP_Z}
""")

            else:
                st.info(f"⚪ **NO TRADE** – Z-Score: {curr_z:.2f} (inside ±{ENTRY_Z})")
                st.caption(f"Strategy type: {strat_name} | Vol: {pair_vol:.1%}")

            # ==================== ENHANCED CHART WITH CLEAR SIGNALS ====================
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=z_score_series.index,
                y=z_score_series,
                name="Z-Score",
                line=dict(color='cyan', width=2)
            ))

            # ENTRY lines
            fig.add_hline(y=ENTRY_Z, line_dash="dash", line_color="red",
                          annotation_text="ENTRY SHORT SPREAD", annotation_position="right",
                          annotation_font=dict(color="red", size=12))
            fig.add_hline(y=-ENTRY_Z, line_dash="dash", line_color="red",
                          annotation_text="ENTRY LONG SPREAD", annotation_position="right",
                          annotation_font=dict(color="red", size=12))

            # EXIT line
            fig.add_hline(y=0, line_dash="dot", line_color="lime",
                          annotation_text="EXIT / COVER POSITION", annotation_position="right",
                          annotation_font=dict(color="lime", size=12))

            # STOP LOSS lines
            fig.add_hline(y=STOP_Z, line_dash="dash", line_color="orange",
                          annotation_text="STOP LOSS – EXIT SHORT", annotation_position="right",
                          annotation_font=dict(color="orange", size=12))
            fig.add_hline(y=-STOP_Z, line_dash="dash", line_color="orange",
                          annotation_text="STOP LOSS – EXIT LONG", annotation_position="right",
                          annotation_font=dict(color="orange", size=12))

            # Current Z marker
            fig.add_trace(go.Scatter(
                x=[z_score_series.index[-1]],
                y=[curr_z],
                mode="markers+text",
                marker=dict(size=14, color="yellow", symbol="star"),
                text=[f"NOW: {curr_z:.2f}"],
                textposition="top center",
                name="Current Z"
            ))

            fig.update_layout(
                template="plotly_dark",
                height=300,
                margin=dict(l=10, r=10, t=30, b=10),
                title="Z-Score with BUY / SELL / COVER / STOP Signals",
                yaxis_title="Z-Score",
                showlegend=True
            )

            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"**Current Z:** {curr_z:.2f} | **Spread Vol:** {pair_vol:.1%} | 60-day rolling")
            st.divider()

    except Exception as e:
        with cols[i % 2]:
            st.error(f"Error processing {a}-{b}: {str(e)}")

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go

st.set_page_config(page_title="Omni-Arb Command Center", layout="wide")
st.title("🚀 Omni-Arb Live Monitor v3.1 – CLEAN EXECUTION")

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

            # === HIGH VISIBILITY EXECUTION BANNER ===
            if abs(curr_z) > ENTRY_Z:
                color_theme = "lightgreen" if curr_z < 0 else "salmon"
                direction = "LONG SPREAD (Buy A / Sell B)" if curr_z < 0 else "SHORT SPREAD (Sell A / Buy B)"
                
                # Green Background Box for Open Trades
                st.markdown(f"""
                    <div style="background-color:{color_theme}; padding:20px; border-radius:10px; border: 2px solid green;">
                        <h2 style="color:black; margin:0;">🔥 SIGNAL DETECTED: {direction}</h2>
                        <p style="color:black; font-size:18px;"><b>PRIMARY TRADE:</b> 100 shares of {a} vs {round(beta, 2)*100} shares of {b}</p>
                        <hr style="border-top: 1px solid black;">
                        <p style="color:black;"><b>Optional Hedge:</b> {strat_name} on {a} ({dte})</p>
                    </div>
                """, unsafe_with_html=True)
                st.write("") # Spacing
            else:
                st.info(f"⚪ **NO TRADE** – Z-Score currently {curr_z:.2f}")

            # ==================== CHART WITH DYNAMIC LABELS ====================
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=z_score_series.index, y=z_score_series, name="Z-Score", line=dict(color='cyan', width=2)))

            # Define dynamic stop level based on position
            current_stop = STOP_Z if curr_z > 0 else -STOP_Z

            # Lines with fixed Z-number labels
            fig.add_hline(y=ENTRY_Z, line_dash="dash", line_color="red", annotation_text=f"ENTRY ({ENTRY_Z})")
            fig.add_hline(y=-ENTRY_Z, line_dash="dash", line_color="red", annotation_text=f"ENTRY (-{ENTRY_Z})")
            fig.add_hline(y=EXIT_Z, line_dash="dot", line_color="lime", annotation_text=f"EXIT ({EXIT_Z})")
            fig.add_hline(y=current_stop, line_dash="dash", line_color="orange", annotation_text=f"STOP ({current_stop})")

            fig.update_layout(
                template="plotly_dark", height=380, margin=dict(l=10, r=10, t=30, b=10),
                title=f"Z-Score Path (Current: {curr_z:.2f})",
                yaxis=dict(range=[-4, 4]) # Keeps charts consistent
            )

            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"**Stats:** Beta: {beta:.2f} | Spread Vol: {pair_vol:.1%}")
            st.divider()

    except Exception as e:
        with cols[i % 2]: st.error(f"Error: {str(e)}")

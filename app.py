import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go

st.set_page_config(page_title="Omni-Arb Command Center", layout="wide")
st.title("🚀 Omni-Arb Live Monitor v3.2 – FIXED")

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

            # === FIXED SIGNAL BANNER ===
            if abs(curr_z) > ENTRY_Z:
                color_theme = "#d4edda" if curr_z < 0 else "#f8d7da" # Soft Green / Soft Red
                border_color = "green" if curr_z < 0 else "red"
                direction = "LONG SPREAD (Buy A / Sell B)" if curr_z < 0 else "SHORT SPREAD (Sell A / Buy B)"
                
                # FIXED: Changed unsafe_with_html to unsafe_allow_html
                st.markdown(f"""
                    <div style="background-color:{color_theme}; padding:20px; border-radius:10px; border: 2px solid {border_color};">
                        <h2 style="color:black; margin:0;">🔥 SIGNAL: {direction}</h2>
                        <p style="color:black; font-size:18px;"><b>PRIMARY:</b> {a} Shares vs {b} Shares (Ratio: 1:{round(beta, 2)})</p>
                        <hr style="border-top: 1px solid black;">
                        <p style="color:black;"><b>HEDGE OPTION:</b> {strat_name} on {a} ({dte})</p>
                    </div>
                """, unsafe_allow_html=True)
                st.write("") 
            else:
                st.info(f"⚪ **NO TRADE** – Z-Score: {curr_z:.2f}")

            # ==================== CHART WITH LABELS ====================
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=z_score_series.index, y=z_score_series, name="Z-Score", line=dict(color='cyan', width=2)))

            # Static markers for Entry/Exit/Stop
            fig.add_hline(y=ENTRY_Z, line_dash="dash", line_color="red", annotation_text=f"ENTRY ({ENTRY_Z})", annotation_position="top right")
            fig.add_hline(y=-ENTRY_Z, line_dash="dash", line_color="red", annotation_text=f"ENTRY (-{ENTRY_Z})", annotation_position="bottom right")
            fig.add_hline(y=0, line_dash="dot", line_color="lime", annotation_text="EXIT (0.0)", annotation_position="top right")
            
            # Dynamic Stop Loss Line
            stop_val = STOP_Z if curr_z > 0 else -STOP_Z
            fig.add_hline(y=stop_val, line_dash="dash", line_color="orange", annotation_text=f"STOP ({stop_val})")

            fig.update_layout(
                template="plotly_dark", height=380, margin=dict(l=10, r=10, t=30, b=10),
                yaxis=dict(range=[-4, 4])
            )

            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"**Current Z:** {curr_z:.2f} | **Spread Vol:** {pair_vol:.1%}")
            st.divider()

    except Exception as e:
        with cols[i % 2]: st.error(f"Error on {a}/{b}: {str(e)}")

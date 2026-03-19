import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import plotly.graph_objects as go
from datetime import datetime

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

st.title("🚀 Omni-Arb Live Monitor v4.0")
st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} EDT")
st.divider()

# ==========================================
# 1. SETUP & PARAMETERS
# ==========================================
PAIRS = [('XOM', 'CVX'), ('V', 'MA'), ('NVDA', 'AMD'), ('KO', 'PEP'), ('GOOGL', 'META')]
ENTRY_Z, EXIT_Z, STOP_Z = 2.25, 0.0, 3.5
VOL_THRESHOLD_FOR_RATIO = 0.20 # 20% volatility required to suggest a ratio spread

@st.cache_data(ttl=3600)
def get_data():
    tickers = list(set([t for p in PAIRS for t in p]))
    data = yf.download(tickers, period="200d", interval="1d")['Close']
    return data

df_raw = get_data()

# ==========================================
# 2. DATA PROCESSING ENGINE
# ==========================================
# We process all data first so we can build the top summary dashboard
processed_data = []
active_signals = []

for ticker_a, ticker_b in PAIRS:
    if ticker_a not in df_raw or ticker_b not in df_raw: continue
    
    pair_df = df_raw[[ticker_a, ticker_b]].dropna()
    if len(pair_df) < 60: continue

    y, x = np.log(pair_df[ticker_a]), sm.add_constant(np.log(pair_df[ticker_b]))
    try:
        model = sm.OLS(y, x).fit()
        beta = model.params[ticker_b]
        
        spread = y - (beta * np.log(pair_df[ticker_b]) + model.params['const'])
        z_score_series = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
        curr_z = z_score_series.iloc[-1]
        pair_vol = spread.pct_change().std() * np.sqrt(252)

        is_active = abs(curr_z) >= ENTRY_Z
        direction = "LONG" if curr_z <= -ENTRY_Z else "SHORT" if curr_z >= ENTRY_Z else "NEUTRAL"
        
        pair_info = {
            'pair': f"{ticker_a} / {ticker_b}",
            'a': ticker_a, 'b': ticker_b,
            'curr_z': curr_z, 'beta': beta, 'vol': pair_vol,
            'z_series': z_score_series,
            'is_active': is_active, 'direction': direction
        }
        processed_data.append(pair_info)
        
        if is_active:
            active_signals.append(pair_info)
            
    except Exception as e:
        st.error(f"Error processing {ticker_a}/{ticker_b}: {str(e)}")

# ==========================================
# 3. TOP DASHBOARD: OPEN TRADES SUMMARY
# ==========================================
st.subheader("⚡ Active Trade Summary")
if not active_signals:
    st.info("No active signals right now. All pairs are within normal trading ranges.")
else:
    sum_cols = st.columns(len(active_signals))
    for i, sig in enumerate(active_signals):
        with sum_cols[i]:
            color = "#28a745" if sig['direction'] == "LONG" else "#dc3545"
            bg_color = "rgba(40, 167, 69, 0.1)" if sig['direction'] == "LONG" else "rgba(220, 53, 69, 0.1)"
            st.markdown(f"""
                <div style="background-color:{bg_color}; padding: 15px; border-radius: 8px; border-top: 4px solid {color}; text-align:center;">
                    <h3 style="margin:0; font-size: 20px; color: {color};">{sig['direction']} {sig['a']}</h3>
                    <p style="margin:5px 0 0 0; font-size: 14px; color: #ccc;">Z-Score: <b>{sig['curr_z']:.2f}</b></p>
                    <p style="margin:0; font-size: 14px; color: #ccc;">vs {sig['b']}</p>
                </div>
            """, unsafe_allow_html=True)
st.divider()

# ==========================================
# 4. INDIVIDUAL CHARTS & TRADE EXECUTION
# ==========================================
chart_cols = st.columns(2)

for i, data in enumerate(processed_data):
    with chart_cols[i % 2]:
        st.markdown(f"### {data['a']} vs {data['b']}")
        
        # --- HIGHLIGHT BOX (ONLY IF ACTIVE) ---
        if data['is_active']:
            if data['direction'] == "LONG":
                box_class = "signal-box-long"
                stock_trade = f"🟢 BUY {data['a']} / 🔴 SELL {data['b']}"
                vert_trade = f"Bull Call Spread on {data['a']} (30-45 DTE)"
                ratio_trade = f"Call Ratio Backspread on {data['a']} (60-70 DTE)"
            else:
                box_class = "signal-box-short"
                stock_trade = f"🔴 SELL {data['a']} / 🟢 BUY {data['b']}"
                vert_trade = f"Bear Put Spread on {data['a']} (30-45 DTE)"
                ratio_trade = f"Put Ratio Backspread on {data['a']} (60-70 DTE)"

            # Dynamic Volatility Check for Ratio Trade
            ratio_html = ""
            if data['vol'] >= VOL_THRESHOLD_FOR_RATIO:
                ratio_html = f"<p style='margin:5px 0 0 0; font-size:15px; color:#ffc107;'><b>3. RATIO SPREAD (High Volatility Play):</b> {ratio_trade}</p>"
            else:
                ratio_html = f"<p style='margin:5px 0 0 0; font-size:13px; color:#6c757d;'><i>*Volatility too low ({data['vol']:.1%}) for a Ratio Backspread. Stick to Stocks or Verticals.</i></p>"

            st.markdown(f"""
                <div class="{box_class}">
                    <h4 style="margin-top:0;">🎯 EXECUTION PLAN</h4>
                    <p style="margin:5px 0; font-size:16px;"><b>1. PURE STOCKS (Primary):</b> {stock_trade} <br>
                    <span style="font-size:13px; color:#aaa;"><i>Ratio: 1 share of {data['a']} per {round(data['beta'], 2)} shares of {data['b']}</i></span></p>
                    <p style="margin:5px 0; font-size:15px;"><b>2. VERTICAL SPREAD (Defined Risk):</b> {vert_trade}</p>
                    {ratio_html}
                </div>
            """, unsafe_allow_html=True)

        # --- PLOTLY CHART ---
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['z_series'].index, y=data['z_series'], name="Z-Score", line=dict(color='#00d1ff', width=2)))

        # Clean Entry/Exit Lines
        fig.add_hline(y=ENTRY_Z, line_dash="dash", line_color="#dc3545", annotation_text=f"SHORT ENTRY (+{ENTRY_Z})", annotation_position="top left")
        fig.add_hline(y=-ENTRY_Z, line_dash="dash", line_color="#28a745", annotation_text=f"LONG ENTRY (-{ENTRY_Z})", annotation_position="bottom left")
        fig.add_hline(y=0, line_width=1, line_color="white", annotation_text="EXIT (0.0)", annotation_position="top right")
        
        # Stop Loss Lines (Faded)
        fig.add_hline(y=STOP_Z, line_dash="dot", line_color="rgba(255, 165, 0, 0.5)", annotation_text=f"STOP LOSS (+{STOP_Z})")
        fig.add_hline(y=-STOP_Z, line_dash="dot", line_color="rgba(255, 165, 0, 0.5)", annotation_text=f"STOP LOSS (-{STOP_Z})")

        fig.update_layout(
            template="plotly_dark", height=350, margin=dict(l=10, r=10, t=10, b=10),
            yaxis=dict(range=[-4, 4], zeroline=False),
            xaxis=dict(showgrid=False)
        )

        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"**Current Z:** {data['curr_z']:.2f} | **Beta:** {data['beta']:.2f} | **Spread Vol:** {data['vol']:.1%}")
        st.markdown("<br><br>", unsafe_allow_html=True) # Spacing between rows

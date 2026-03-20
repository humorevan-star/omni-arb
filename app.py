# =============================================================================
# OMNI-ARB v6.0  |  Statistical Arbitrage Terminal
# State-machine logic: Entry Hysteresis / Toggle Logic
# Focus: Execution-Centric Signals (Stocks & Options)
# Enhanced: Rich UX, Active-Trade Zone Charts, & Full Backtester
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
import math as _math

# =============================================================================
# 0. CONFIG & THEME
# =============================================================================
st.set_page_config(
    page_title="Omni-Arb v6.0",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
    .main { background-color: #0b0e14; color: #e1e1e1; }
    .stPlotlyChart { background-color: #0b0e14; border-radius: 8px; }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.5px; }
    code { color: #00d1ff !important; background: rgba(0,209,255,0.08) !important;
           padding: 2px 6px !important; border-radius: 3px !important; font-size: 12px !important; }
    .stSpinner > div { border-top-color: #00ffcc !important; }
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

# Backtest parameters
BT_CAPITAL       = 2000    
BT_MAX_OPEN      = 3       
BT_MAX_HOLD_DAYS = 21      
BT_LOOKBACK_DAYS = 730     
BT_ALLOC_PER     = BT_CAPITAL / len(PAIRS)

# =============================================================================
# 2. SIZING & REBALANCE
# =============================================================================
def compute_legs(sig, capital=STARTING_CAPITAL):
    """Medallion Delta-Neutral Sizing."""
    price_a = float(sig["price_a"])
    price_b = float(sig["price_b"])
    beta    = max(abs(float(sig["beta"])), 0.01)

    dollar_a = capital / (1.0 + beta)
    dollar_b = capital - dollar_a
    shares_a = max(0.1, round(dollar_a / price_a, 1))
    shares_b = max(0.1, round(dollar_b / price_b, 1))

    notional_a = round(shares_a * price_a, 2)
    notional_b = round(shares_b * price_b, 2)
    total_cost = round(notional_a + notional_b, 2)
    
    risk_imbalance = round(notional_a - (notional_b / beta), 2)
    imbalance      = round(notional_a - notional_b, 2)
    
    pnl = pnl_a = pnl_b = None
    ot = sig.get("open_trade")
    if ot and ot.get("entry_price_a"):
        ep_a, ep_b = float(ot["entry_price_a"]), float(ot["entry_price_b"])
        is_long = ot["direction"] == "LONG"
        _sa = float(ot.get("entry_shares_a", shares_a))
        _sb = float(ot.get("entry_shares_b", shares_b))
        
        # Dollar P&L
        if is_long:
            pnl_a, pnl_b = (price_a - ep_a) * _sa, -(price_b - ep_b) * _sb
        else:
            pnl_a, pnl_b = -(price_a - ep_a) * _sa, (price_b - ep_b) * _sb
            
        # Spread P&L = Log-return based Medallion method
        notional_a_entry = _sa * ep_a
        log_spread_entry = _math.log(ep_a) - beta * _math.log(ep_b)
        log_spread_now   = _math.log(price_a) - beta * _math.log(price_b)
        pnl = round((log_spread_now - log_spread_entry) * notional_a_entry * (1 if is_long else -1), 2)

    return {
        "shares_a": shares_a, "shares_b": shares_b, "ratio": f"{shares_a}:{shares_b}",
        "notional_a": notional_a, "notional_b": notional_b, "total_cost": total_cost,
        "risk_imbalance": risk_imbalance, "imbalance": imbalance,
        "pnl": pnl, "pnl_a": pnl_a, "pnl_b": pnl_b, "entry_pa": ot.get("entry_price_a") if ot else None,
        "entry_pb": ot.get("entry_price_b") if ot else None
    }

def get_rebalance_instructions(sig):
    ot = sig.get("open_trade")
    if not ot or not ot.get("entry_date"):
        return {"status": "STABLE", "reason": "No open trade", "days_in": 0}
    days_in = (pd.Timestamp.now().normalize() - pd.Timestamp(ot["entry_date"]).normalize()).days
    if days_in <= 0 or days_in % 2 != 0:
        return {"status": "STABLE", "reason": f"Day {days_in} - Monitoring", "days_in": days_in}
    
    # Check for beta drift
    current_beta = max(abs(float(sig["beta"])), 0.01)
    ideal_sb = max(0.1, round((STARTING_CAPITAL - (STARTING_CAPITAL / (1.0 + current_beta))) / float(sig["price_b"]), 1))
    locked_sb = float(ot.get("entry_shares_b") or 0)
    diff = round(ideal_sb - locked_sb, 1)
    
    if abs(diff) < 0.1:
        return {"status": "STABLE", "reason": "Hedge stable", "days_in": days_in}
    
    return {"status": "REBALANCE", "action": "BUY" if diff > 0 else "SELL", "qty": abs(diff), 
            "ticker": sig["b"], "reason": f"Beta drifted. Adjust {sig['b']} by {diff:+.1f} shs.", "days_in": days_in}

# =============================================================================
# 3. DATA ENGINE (STATE-MACHINE)
# =============================================================================
@st.cache_data(ttl=3600)
def get_market_data():
    tickers = list(set([t for p in PAIRS for t in p]))
    return yf.download(tickers, period="750d", interval="1d")["Close"]

def process_pairs(df_raw):
    processed, active = [], []
    for ticker_a, ticker_b in PAIRS:
        if ticker_a not in df_raw.columns or ticker_b not in df_raw.columns: continue
        pair_df = df_raw[[ticker_a, ticker_b]].dropna()
        y, x = np.log(pair_df[ticker_a]), sm.add_constant(np.log(pair_df[ticker_b]))
        model = RollingOLS(y, x, window=ROLLING_WINDOW).fit()
        betas, consts = model.params[ticker_b], model.params["const"]
        spread = (y - (betas * np.log(pair_df[ticker_b]) + consts)).dropna()
        z_series = ((spread - spread.rolling(ROLLING_WINDOW).mean()) / spread.rolling(ROLLING_WINDOW).std()).dropna()
        
        # State machine tracking through history
        in_open, open_dir, open_z, open_dt = False, None, None, None
        for dt, z in z_series.items():
            if not in_open:
                if z <= -ENTRY_Z: in_open, open_dir, open_z, open_dt = True, "LONG", z, dt
                elif z >= ENTRY_Z: in_open, open_dir, open_z, open_dt = True, "SHORT", z, dt
            else:
                if (open_dir == "LONG" and z >= 0) or (open_dir == "SHORT" and z <= 0) or abs(z) >= STOP_Z:
                    in_open = False
        
        open_trade = {
            "direction": open_dir, "entry_z": open_z, "entry_date": open_dt,
            "entry_price_a": pair_df[ticker_a].loc[open_dt] if in_open else None,
            "entry_price_b": pair_df[ticker_b].loc[open_dt] if in_open else None,
            "entry_shares_a": max(0.1, round((STARTING_CAPITAL / (1.0 + abs(betas.loc[open_dt]))) / pair_df[ticker_a].loc[open_dt], 1)) if in_open else None,
            "entry_shares_b": max(0.1, round((STARTING_CAPITAL - (STARTING_CAPITAL / (1.0 + abs(betas.loc[open_dt])))) / pair_df[ticker_b].loc[open_dt], 1)) if in_open else None
        } if in_open else None

        info = {
            "pair": f"{ticker_a}/{ticker_b}", "a": ticker_a, "b": ticker_b, "curr_z": z_series.iloc[-1], 
            "beta": betas.iloc[-1], "z_series": z_series, "price_a": pair_df[ticker_a].iloc[-1], 
            "price_b": pair_df[ticker_b].iloc[-1], "open_trade": open_trade, "direction": open_dir or "NEUTRAL",
            "is_cointegrated": adfuller(spread)[1] < 0.05, "adf_pval": adfuller(spread)[1],
            "pair_df": pair_df, "betas_series": betas
        }
        processed.append(info)
        if in_open: active.append(info)
    return processed, active

# =============================================================================
# 4. HTML HELPERS & UI
# =============================================================================
def _row(label, value, val_color="#e8eaf0"):
    return f'<div style="display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.04);"><span style="font-size:11px;color:#4a5568;font-family:monospace;">{label}</span><span style="font-family:monospace;font-size:12px;font-weight:500;color:{val_color};">{value}</span></div>'

def render_trade_card(sig):
    is_long = sig["direction"] == "LONG"
    accent = "#00ffcc" if is_long else "#ff4b4b"
    legs = compute_legs(sig)
    rb = get_rebalance_instructions(sig)
    
    html = f"""
    <div style="background:rgba({('0,255,204' if is_long else '255,75,75')},0.04);border:1px solid rgba({('0,255,204' if is_long else '255,75,75')},0.25);border-top:3px solid {accent};padding:20px;border-radius:6px;margin-bottom:20px;">
        <div style="display:flex;justify-content:space-between;">
            <div><h2 style="margin:0;color:{accent};">{sig['pair']}</h2><span style="font-size:12px;color:#8892a4;">{sig['direction']} SPREAD OPEN</span></div>
            <div style="text-align:right;"><p style="margin:0;font-size:10px;color:#4a5568;">Z-SCORE NOW</p><p style="margin:0;font-size:30px;font-weight:600;color:{accent};">{sig['curr_z']:.2f}</p></div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:20px;">
            <div style="background:rgba(0,0,0,0.2);padding:15px;border-radius:4px;">
                <p style="margin:0 0 10px;font-size:10px;color:#4a9eff;letter-spacing:0.1em;">EXECUTION BREAKDOWN</p>
                {_row(f"{('BUY' if is_long else 'SELL')} {sig['a']}", f"{legs['shares_a']} shs @ ${sig['price_a']:.2f}", accent)}
                {_row(f"{('SELL' if is_long else 'BUY')} {sig['b']}", f"{legs['shares_b']} shs @ ${sig['price_b']:.2f}", ('#ff4b4b' if is_long else '#00ffcc'))}
                {_row("Total Deployed", f"${legs['total_cost']:,.0f}")}
                {_row("Unrealised P&L", f"${legs['pnl']:+,.2f}", ('#00d4a0' if legs['pnl'] >= 0 else '#f56565'))}
            </div>
            <div style="background:rgba(0,0,0,0.2);padding:15px;border-radius:4px;">
                <p style="margin:0 0 10px;font-size:10px;color:#a78bfa;letter-spacing:0.1em;">HEDGE MONITOR</p>
                {_row("Status", rb['status'], ('#f5a623' if rb['status'] == 'REBALANCE' else '#00d4a0'))}
                <p style="font-size:11px;color:#8892a4;margin-top:10px;">{rb['reason']}</p>
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

# =============================================================================
# 5. CHARTS
# =============================================================================
def render_pair_chart(p):
    z_data, ot = p["z_series"], p.get("open_trade")
    fig = go.Figure()
    
    # Active trade zone shading
    if ot:
        y0, y1 = min(float(ot["entry_z"]), 0), max(float(ot["entry_z"]), 0)
        fig.add_hrect(y0=y0, y1=y1, fillcolor="rgba(0,255,204,0.05)" if ot["direction"]=="LONG" else "rgba(255,75,75,0.05)", line_width=0)

    fig.add_trace(go.Scatter(x=z_data.index, y=z_data, line=dict(color="#00d1ff", width=2), name="Z-Score"))
    
    if ot:
        color = "#00ffcc" if ot["direction"]=="LONG" else "#ff4b4b"
        fig.add_trace(go.Scatter(x=[ot["entry_date"]], y=[ot["entry_z"]], mode="markers", 
                                 marker=dict(color=color, size=14, symbol="star", line=dict(color="white", width=1))))
    
    fig.add_hline(y=ENTRY_Z, line_dash="dash", line_color="#ff4b4b", opacity=0.3)
    fig.add_hline(y=-ENTRY_Z, line_dash="dash", line_color="#00ffcc", opacity=0.3)
    fig.add_hline(y=0, line_color="white", line_width=1, opacity=0.2)
    fig.update_layout(template="plotly_dark", height=350, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
                      xaxis=dict(showgrid=False), yaxis=dict(showgrid=False))
    return fig

# =============================================================================
# 6. BACKTEST ENGINE
# =============================================================================
def run_backtest(all_pairs_data):
    # Simplified simulation for brevity but logic is event-driven
    n = len(all_pairs_data)
    alloc = BT_CAPITAL / n
    trades = []
    
    for p in all_pairs_data:
        z = p["z_series"]
        pdf = p["pair_df"]
        beta_s = p["betas_series"]
        in_trade, entry_z, entry_dt, direction = False, 0, None, None
        
        for dt, val in z.items():
            if not in_trade:
                if val <= -ENTRY_Z: in_trade, entry_z, entry_dt, direction = True, val, dt, "LONG"
                elif val >= ENTRY_Z: in_trade, entry_z, entry_dt, direction = True, val, dt, "SHORT"
            else:
                # Exit Logic
                days_held = (dt - entry_dt).days
                if (direction == "LONG" and val >= 0) or (direction == "SHORT" and val <= 0) or abs(val) >= STOP_Z or days_held >= BT_MAX_HOLD_DAYS:
                    ep_a, ep_b = pdf[p['a']].loc[entry_dt], pdf[p['b']].loc[entry_dt]
                    cp_a, cp_b = pdf[p['a']].loc[dt], pdf[p['b']].loc[dt]
                    beta = abs(beta_s.loc[entry_dt])
                    
                    # P&L Calculation
                    d_a = alloc / (1.0 + beta)
                    sa = d_a / ep_a
                    
                    log_entry = _math.log(ep_a) - beta * _math.log(ep_b)
                    log_exit = _math.log(cp_a) - beta * _math.log(cp_b)
                    pnl = (log_exit - log_entry) * (sa * ep_a) * (1 if direction == "LONG" else -1)
                    
                    trades.append({"pnl": pnl, "date": dt, "pair": p['pair']})
                    in_trade = False
                    
    if not trades: return None
    
    df_trades = pd.DataFrame(trades).sort_values("date")
    df_trades["cum_pnl"] = df_trades["pnl"].cumsum() + BT_CAPITAL
    return df_trades

def render_backtest(all_pairs):
    st.divider()
    st.header("Strategic Backtest")
    bt_data = run_backtest(all_pairs)
    
    if bt_data is not None:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bt_data["date"], y=bt_data["cum_pnl"], fill='tozeroy', line=dict(color="#00d4a0", width=3)))
        fig.update_layout(template="plotly_dark", height=400, title="Equity Curve ($2,000 Portfolio)",
                          xaxis=dict(showgrid=False), yaxis=dict(showgrid=False, tickprefix="$"))
        st.plotly_chart(fig, use_container_width=True)
        
        # Stats
        total_pnl = bt_data['pnl'].sum()
        win_rate = (bt_data['pnl'] > 0).mean() * 100
        k = st.columns(4)
        k[0].metric("Total P&L", f"${total_pnl:,.2f}")
        k[1].metric("Win Rate", f"{win_rate:.1f}%")
        k[2].metric("Trades", len(bt_data))
        k[3].metric("Final Equity", f"${bt_data['cum_pnl'].iloc[-1]:,.2f}")
    else:
        st.warning("No historical trades met the criteria in the lookback window.")

# =============================================================================
# 10. MAIN
# =============================================================================
def main():
    st.title("Omni-Arb Terminal v6.0")
    st.caption(f"Universe: S&P 500 Pairs | {datetime.now().strftime('%b %d %H:%M ET')}")
    
    data_raw = get_market_data()
    all_pairs, active_pairs = process_pairs(data_raw)
    
    st.subheader("Active Signals")
    if active_pairs:
        for sig in active_pairs: render_trade_card(sig)
    else:
        st.write("Monitoring market for cointegration deviations...")

    st.divider()
    st.subheader("Pair Watchlist")
    cols = st.columns(2)
    for i, p in enumerate(all_pairs):
        with cols[i % 2]:
            st.markdown(f"**{p['pair']}** | β: {p['beta']:.2f}")
            st.plotly_chart(render_pair_chart(p), use_container_width=True)
            
    render_backtest(all_pairs)

if __name__ == "__main__":
    main()

# =============================================================================
# OMNI-ARB v5.3  |  Statistical Arbitrage Terminal
# Adds: 2-year backtest, P&L, APY, win/loss stats per pair + portfolio
# =============================================================================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime

# =============================================================================
# 0. CONFIG & THEME
# =============================================================================
st.set_page_config(page_title="Omni-Arb v5.3", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
    <style>
    .main { background-color: #0b0e14; color: #e1e1e1; }
    .stPlotlyChart { background-color: #0b0e14; border-radius: 10px; }
    [data-testid="stHeader"] { background: rgba(0,0,0,0); }
    h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.5px; }
    </style>
    """, unsafe_allow_html=True)

# =============================================================================
# 1. PARAMETERS
# =============================================================================
PAIRS = [('XOM', 'CVX'), ('V', 'MA'), ('NVDA', 'AMD'), ('KO', 'PEP'), ('GOOGL', 'META')]
ENTRY_Z        = 2.25
STOP_Z         = 3.5
EXIT_Z         = 0.0
ROLLING_WINDOW = 60
STARTING_CAPITAL = 1000
BT_CAPITAL     = 2000    # backtest starting capital per pair
BT_PERIOD      = "730d"  # ~2 years


# =============================================================================
# 2. SIZING
# =============================================================================
def compute_legs(sig, capital=STARTING_CAPITAL):
    target_per_leg = capital / 2
    shares_a = max(1, int(target_per_leg / sig["price_a"]))
    shares_b = max(1, int(shares_a * sig["beta"]))
    return {
        "shares_a": shares_a,  "notional_a": shares_a * sig["price_a"],
        "shares_b": shares_b,  "notional_b": shares_b * sig["price_b"],
        "total_cost": shares_a * sig["price_a"] + shares_b * sig["price_b"],
    }


# =============================================================================
# 3. DATA ENGINE
# =============================================================================
@st.cache_data(ttl=3600)
def get_market_data():
    tickers = list(set([t for p in PAIRS for t in p]))
    data = yf.download(tickers, period="750d", interval="1d")["Close"]
    return data


def process_pairs(df_raw):
    processed, active = [], []
    for ticker_a, ticker_b in PAIRS:
        if ticker_a not in df_raw.columns or ticker_b not in df_raw.columns:
            continue
        pair_df = df_raw[[ticker_a, ticker_b]].dropna()
        y = np.log(pair_df[ticker_a])
        x = sm.add_constant(np.log(pair_df[ticker_b]))

        model  = RollingOLS(y, x, window=ROLLING_WINDOW).fit()
        betas  = model.params[ticker_b]
        consts = model.params["const"]
        spread = y - (betas * np.log(pair_df[ticker_b]) + consts)
        spread = spread.dropna()

        z_series = (
            (spread - spread.rolling(ROLLING_WINDOW).mean())
            / spread.rolling(ROLLING_WINDOW).std()
        )
        curr_z = z_series.iloc[-1]

        long_entries  = z_series[(z_series <= -ENTRY_Z) & (z_series.shift(1) > -ENTRY_Z)]
        short_entries = z_series[(z_series >= ENTRY_Z)  & (z_series.shift(1) < ENTRY_Z)]
        exits = z_series[
            ((z_series >= 0) & (z_series.shift(1) < 0))
            | ((z_series <= 0) & (z_series.shift(1) > 0))
        ]

        adf_pval = adfuller(spread)[1]
        info = {
            "pair": f"{ticker_a}/{ticker_b}",
            "a": ticker_a, "b": ticker_b,
            "price_a": pair_df[ticker_a].iloc[-1],
            "price_b": pair_df[ticker_b].iloc[-1],
            "curr_z": curr_z,
            "beta": betas.iloc[-1],
            "z_series": z_series,
            "spread": spread,
            "pair_df": pair_df,
            "betas_series": betas,
            "consts_series": consts,
            "long_spots": long_entries,
            "short_spots": short_entries,
            "exit_spots": exits,
            "is_cointegrated": adf_pval < 0.05,
            "adf_pval": adf_pval,
            "direction": (
                "LONG"  if curr_z <= -ENTRY_Z else
                "SHORT" if curr_z >=  ENTRY_Z else
                "NEUTRAL"
            ),
        }
        processed.append(info)
        if info["direction"] != "NEUTRAL":
            active.append(info)
    return processed, active


# =============================================================================
# 4. BACKTEST ENGINE
# =============================================================================
def run_backtest(pair_info: dict, capital: float = BT_CAPITAL) -> dict:
    """
    Event-driven backtest on z-score signals.
    Entry: |z| crosses ENTRY_Z
    Exit:  z reverts to EXIT_Z (zero cross) OR hits STOP_Z
    Returns per-trade log + equity curve.
    """
    z   = pair_info["z_series"].dropna()
    pdf = pair_info["pair_df"].reindex(z.index).dropna()
    beta_s  = pair_info["betas_series"].reindex(z.index).ffill()
    ticker_a, ticker_b = pair_info["a"], pair_info["b"]

    # Align all series to common index
    common = z.index.intersection(pdf.index).intersection(beta_s.index)
    z      = z.loc[common]
    pdf    = pdf.loc[common]
    beta_s = beta_s.loc[common]

    # Only backtest on last ~2 years
    cutoff = z.index[-1] - pd.DateOffset(days=730)
    z      = z[z.index >= cutoff]
    pdf    = pdf[pdf.index >= cutoff]
    beta_s = beta_s[beta_s.index >= cutoff]

    trades     = []
    equity     = [capital]
    eq_dates   = [z.index[0]]
    in_trade   = False
    direction  = None
    entry_idx  = None
    entry_pa   = None
    entry_pb   = None
    entry_beta = None
    shares_a   = 0
    shares_b   = 0

    for i in range(1, len(z)):
        date   = z.index[i]
        curr   = z.iloc[i]
        prev   = z.iloc[i - 1]
        pa     = pdf[ticker_a].iloc[i]
        pb     = pdf[ticker_b].iloc[i]
        beta_v = beta_s.iloc[i]
        current_equity = equity[-1]

        if not in_trade:
            # LONG entry: z crosses below -ENTRY_Z
            if prev > -ENTRY_Z and curr <= -ENTRY_Z:
                target = current_equity / 2
                shares_a = max(1, int(target / pa))
                shares_b = max(1, int(shares_a * abs(beta_v)))
                in_trade   = True
                direction  = "LONG"
                entry_idx  = date
                entry_pa   = pa
                entry_pb   = pb
                entry_beta = beta_v

            # SHORT entry: z crosses above +ENTRY_Z
            elif prev < ENTRY_Z and curr >= ENTRY_Z:
                target = current_equity / 2
                shares_a = max(1, int(target / pa))
                shares_b = max(1, int(shares_a * abs(beta_v)))
                in_trade   = True
                direction  = "SHORT"
                entry_idx  = date
                entry_pa   = pa
                entry_pb   = pb
                entry_beta = beta_v

        else:
            hit_exit = (
                (direction == "LONG"  and prev < 0 and curr >= 0) or
                (direction == "SHORT" and prev > 0 and curr <= 0)
            )
            hit_stop = abs(curr) >= STOP_Z

            if hit_exit or hit_stop:
                # P&L calculation (dollar-neutral spread)
                if direction == "LONG":
                    pnl = (shares_a * (pa - entry_pa)) - (shares_b * (pb - entry_pb))
                else:
                    pnl = (shares_a * (entry_pa - pa)) - (shares_b * (entry_pb - pb))

                new_equity = current_equity + pnl
                equity.append(new_equity)
                eq_dates.append(date)

                hold_days = (date - entry_idx).days
                trades.append({
                    "entry_date":  entry_idx,
                    "exit_date":   date,
                    "direction":   direction,
                    "pnl":         pnl,
                    "pnl_pct":     pnl / current_equity * 100,
                    "hold_days":   hold_days,
                    "exit_reason": "STOP" if hit_stop else "EXIT",
                    "entry_z":     z.loc[entry_idx] if entry_idx in z.index else float("nan"),
                    "exit_z":      curr,
                })
                in_trade = False

        # Append current equity if no trade closed today
        if not equity or eq_dates[-1] != date:
            equity.append(equity[-1])
            eq_dates.append(date)

    # Compute stats
    if not trades:
        return {
            "trades": [], "equity_curve": pd.Series(equity, index=eq_dates),
            "total_pnl": 0, "win_rate": 0, "num_trades": 0,
            "num_wins": 0, "num_losses": 0, "apy": 0,
            "max_drawdown": 0, "avg_hold": 0, "best_trade": 0, "worst_trade": 0,
        }

    trade_df  = pd.DataFrame(trades)
    eq_series = pd.Series(equity, index=eq_dates).groupby(level=0).last()

    total_pnl  = trade_df["pnl"].sum()
    num_wins   = int((trade_df["pnl"] > 0).sum())
    num_losses = int((trade_df["pnl"] <= 0).sum())
    win_rate   = num_wins / len(trade_df) * 100 if len(trade_df) else 0

    # APY
    days_span = max((eq_series.index[-1] - eq_series.index[0]).days, 1)
    final_eq  = eq_series.iloc[-1]
    apy = ((final_eq / capital) ** (365 / days_span) - 1) * 100

    # Max drawdown
    roll_max  = eq_series.cummax()
    drawdown  = (eq_series - roll_max) / roll_max * 100
    max_dd    = drawdown.min()

    return {
        "trades":       trades,
        "trade_df":     trade_df,
        "equity_curve": eq_series,
        "total_pnl":    total_pnl,
        "final_equity": final_eq,
        "win_rate":     win_rate,
        "num_trades":   len(trade_df),
        "num_wins":     num_wins,
        "num_losses":   num_losses,
        "apy":          apy,
        "max_drawdown": max_dd,
        "avg_hold":     trade_df["hold_days"].mean(),
        "best_trade":   trade_df["pnl"].max(),
        "worst_trade":  trade_df["pnl"].min(),
    }


# =============================================================================
# 5. UI HELPERS
# =============================================================================
def render_trade_card(sig):
    is_long = sig["direction"] == "LONG"
    accent  = "#00ffcc" if is_long else "#ff4b4b"
    bg      = "rgba(0, 255, 204, 0.05)" if is_long else "rgba(255, 75, 75, 0.05)"
    legs    = compute_legs(sig)

    leg1_verb = "BUY"  if is_long else "SELL"
    leg2_verb = "SELL" if is_long else "BUY"
    leg1_col  = "#00ffcc" if is_long else "#ff4b4b"
    leg2_col  = "#ff4b4b" if is_long else "#00ffcc"
    action_text = "BUY THE SPREAD"  if is_long else "SELL THE SPREAD"
    state_text  = "Lagging"         if is_long else "Overextended"

    st.markdown(
        '<div style="background:' + bg + ';border:1px solid ' + accent + ';padding:20px;'
        'border-radius:8px;margin-bottom:20px;">'
        '<div style="display:flex;justify-content:space-between;align-items:center;">'
        '<h2 style="margin:0;color:' + accent + ';font-size:24px;font-family:monospace;">'
        + sig["a"] + " / " + sig["b"] + "</h2>"
        '<div style="text-align:right;">'
        '<span style="font-size:12px;color:#8892a4;">CURRENT Z-SCORE</span><br>'
        '<b style="font-size:20px;color:' + accent + ';">' + f"{sig['curr_z']:.2f}" + "</b>"
        "</div></div>"
        '<hr style="border:0;border-top:1px solid rgba(255,255,255,0.1);margin:15px 0;">'
        '<div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">'
        "<div>"
        '<p style="font-size:11px;color:#4a5568;margin-bottom:8px;text-transform:uppercase;">Action Required</p>'
        '<b style="font-size:16px;color:#fff;">' + action_text + "</b><br>"
        '<span style="font-size:13px;color:#8892a4;">' + sig["a"] + " is " + state_text + "</span>"
        "</div>"
        '<div style="background:rgba(0,0,0,0.2);padding:10px;border-radius:4px;">'
        '<p style="font-size:10px;color:#4a5568;margin:0;">EXECUTION GUIDE ($1K)</p>'
        '<div style="display:flex;justify-content:space-between;margin-top:5px;font-family:monospace;">'
        '<span style="color:' + leg1_col + ';">' + leg1_verb + " " + str(legs["shares_a"]) + " " + sig["a"] + "</span>"
        '<span style="color:' + leg2_col + ';">' + leg2_verb + " " + str(legs["shares_b"]) + " " + sig["b"] + "</span>"
        "</div></div></div></div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# 6. BACKTEST DASHBOARD RENDERER
# =============================================================================
def render_backtest_section(all_pairs: list):
    st.divider()
    st.markdown(
        '<p style="font-family:monospace;font-size:11px;letter-spacing:0.14em;'
        'color:#4a5568;text-transform:uppercase;margin-bottom:4px;">2-Year Historical Simulation</p>',
        unsafe_allow_html=True,
    )
    st.subheader("📈 Backtest Results  |  $2,000 Starting Capital Per Pair")

    # Run all backtests
    results = {}
    for p in all_pairs:
        results[p["pair"]] = run_backtest(p, capital=BT_CAPITAL)

    # ── Portfolio equity curve (sum of all pairs) ─────────────
    all_curves = []
    for pair_key, r in results.items():
        eq = r["equity_curve"]
        if len(eq) > 1:
            # Normalise to P&L contribution (delta from starting capital)
            all_curves.append(eq - BT_CAPITAL)

    if all_curves:
        combined_index = all_curves[0].index
        for c in all_curves[1:]:
            combined_index = combined_index.union(c.index)

        portfolio_pnl = sum(
            c.reindex(combined_index).ffill().fillna(0) for c in all_curves
        )
        portfolio_equity = portfolio_pnl + BT_CAPITAL * len(all_pairs)
    else:
        portfolio_equity = pd.Series([BT_CAPITAL * len(all_pairs)])

    # ── Summary stats ─────────────────────────────────────────
    total_trades  = sum(r["num_trades"]  for r in results.values())
    total_wins    = sum(r["num_wins"]    for r in results.values())
    total_losses  = sum(r["num_losses"]  for r in results.values())
    total_pnl     = sum(r["total_pnl"]   for r in results.values())
    total_capital = BT_CAPITAL * len(all_pairs)
    port_final    = portfolio_equity.iloc[-1] if len(portfolio_equity) > 0 else total_capital
    days_span     = max((portfolio_equity.index[-1] - portfolio_equity.index[0]).days, 1) if len(portfolio_equity) > 1 else 365
    port_apy      = ((port_final / total_capital) ** (365 / days_span) - 1) * 100
    port_wr       = (total_wins / total_trades * 100) if total_trades else 0

    # Portfolio drawdown
    roll_max = portfolio_equity.cummax()
    port_dd  = ((portfolio_equity - roll_max) / roll_max * 100).min()

    pnl_color  = "#00d4a0" if total_pnl >= 0 else "#f56565"
    apy_color  = "#00d4a0" if port_apy  >= 0 else "#f56565"
    dd_color   = "#f5a623" if port_dd   > -10 else "#f56565"

    # ── KPI strip ─────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)

    def kpi(col, label, value, color="#e8eaf0"):
        col.markdown(
            '<div style="background:#111318;padding:14px 16px;border-radius:4px;'
            'border:1px solid rgba(255,255,255,0.07);text-align:center;">'
            '<p style="margin:0 0 4px;font-size:9px;color:#4a5568;font-family:monospace;'
            'text-transform:uppercase;letter-spacing:0.1em;">' + label + "</p>"
            '<p style="margin:0;font-family:monospace;font-size:20px;font-weight:600;color:'
            + color + ';">' + value + "</p></div>",
            unsafe_allow_html=True,
        )

    kpi(k1, "Total P&L",      f"${total_pnl:+,.0f}",    pnl_color)
    kpi(k2, "Portfolio APY",  f"{port_apy:+.1f}%",       apy_color)
    kpi(k3, "Win Rate",       f"{port_wr:.0f}%",         "#e8c96d")
    kpi(k4, "Total Trades",   str(total_trades),          "#e8eaf0")
    kpi(k5, "W / L",          f"{total_wins} / {total_losses}", "#e8eaf0")
    kpi(k6, "Max Drawdown",   f"{port_dd:.1f}%",         dd_color)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Portfolio equity chart ────────────────────────────────
    fig_port = go.Figure()
    fig_port.add_trace(go.Scatter(
        x=portfolio_equity.index, y=portfolio_equity,
        fill="tozeroy",
        fillcolor="rgba(0,212,160,0.07)",
        line=dict(color="#00d4a0", width=2.5),
        name="Portfolio Equity",
    ))
    fig_port.add_hline(
        y=total_capital,
        line=dict(color="rgba(255,255,255,0.2)", width=1, dash="dot"),
        annotation_text="Starting Capital",
        annotation_font_color="#4a5568",
        annotation_position="top left",
    )
    fig_port.update_layout(
        template="plotly_dark", height=220,
        paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=False, zeroline=False, tickprefix="$"),
        font=dict(family="IBM Plex Mono"),
    )
    st.plotly_chart(fig_port, use_container_width=True)

    # ── Per-pair breakdown table ──────────────────────────────
    st.markdown(
        '<p style="font-family:monospace;font-size:10px;letter-spacing:0.12em;'
        'color:#4a5568;text-transform:uppercase;margin:8px 0 12px;">Per-Pair Performance</p>',
        unsafe_allow_html=True,
    )

    pair_cols = st.columns(len(all_pairs))
    for i, p in enumerate(all_pairs):
        r   = results[p["pair"]]
        col = pair_cols[i]

        pnl_c  = "#00d4a0" if r["total_pnl"] >= 0 else "#f56565"
        apy_c  = "#00d4a0" if r["apy"]       >= 0 else "#f56565"
        coint_c= "#00d4a0" if p["is_cointegrated"] else "#f5a623"

        col.markdown(
            '<div style="background:#111318;padding:14px;border-radius:4px;'
            'border:1px solid rgba(255,255,255,0.07);height:100%;">'

            # Header
            '<p style="margin:0 0 10px;font-family:monospace;font-size:13px;'
            'font-weight:600;color:#e8eaf0;">' + p["pair"] + "</p>"

            # P&L
            '<div style="display:flex;justify-content:space-between;margin-bottom:5px;">'
            '<span style="font-size:10px;color:#4a5568;font-family:monospace;">P&amp;L</span>'
            '<span style="font-family:monospace;font-size:12px;color:' + pnl_c + ';">'
            + f"${r['total_pnl']:+,.0f}" + "</span></div>"

            # APY
            '<div style="display:flex;justify-content:space-between;margin-bottom:5px;">'
            '<span style="font-size:10px;color:#4a5568;font-family:monospace;">APY</span>'
            '<span style="font-family:monospace;font-size:12px;color:' + apy_c + ';">'
            + f"{r['apy']:+.1f}%" + "</span></div>"

            # Win rate
            '<div style="display:flex;justify-content:space-between;margin-bottom:5px;">'
            '<span style="font-size:10px;color:#4a5568;font-family:monospace;">Win Rate</span>'
            '<span style="font-family:monospace;font-size:12px;color:#e8c96d;">'
            + f"{r['win_rate']:.0f}%" + "</span></div>"

            # Trades
            '<div style="display:flex;justify-content:space-between;margin-bottom:5px;">'
            '<span style="font-size:10px;color:#4a5568;font-family:monospace;">Trades</span>'
            '<span style="font-family:monospace;font-size:12px;color:#e8eaf0;">'
            + f"{r['num_trades']} ({r['num_wins']}W/{r['num_losses']}L)" + "</span></div>"

            # Avg hold
            '<div style="display:flex;justify-content:space-between;margin-bottom:5px;">'
            '<span style="font-size:10px;color:#4a5568;font-family:monospace;">Avg Hold</span>'
            '<span style="font-family:monospace;font-size:12px;color:#e8eaf0;">'
            + (f"{r['avg_hold']:.0f}d" if r["num_trades"] else "–") + "</span></div>"

            # Max DD
            '<div style="display:flex;justify-content:space-between;margin-bottom:5px;">'
            '<span style="font-size:10px;color:#4a5568;font-family:monospace;">Max DD</span>'
            '<span style="font-family:monospace;font-size:12px;color:#f5a623;">'
            + f"{r['max_drawdown']:.1f}%" + "</span></div>"

            # Best / worst
            '<div style="border-top:1px solid rgba(255,255,255,0.07);margin-top:8px;padding-top:8px;">'
            '<div style="display:flex;justify-content:space-between;margin-bottom:4px;">'
            '<span style="font-size:10px;color:#4a5568;font-family:monospace;">Best</span>'
            '<span style="font-family:monospace;font-size:11px;color:#00d4a0;">'
            + (f"${r['best_trade']:+,.0f}" if r["num_trades"] else "–") + "</span></div>"
            '<div style="display:flex;justify-content:space-between;">'
            '<span style="font-size:10px;color:#4a5568;font-family:monospace;">Worst</span>'
            '<span style="font-family:monospace;font-size:11px;color:#f56565;">'
            + (f"${r['worst_trade']:+,.0f}" if r["num_trades"] else "–") + "</span></div>"
            "</div>"

            # Cointegration badge
            '<div style="margin-top:10px;text-align:center;">'
            '<span style="font-size:9px;font-family:monospace;padding:2px 8px;border-radius:2px;'
            'color:' + coint_c + ';border:1px solid ' + coint_c + ';opacity:0.8;">'
            + ("COINTEGRATED" if p["is_cointegrated"] else "DRIFTING") + "</span></div>"

            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Per-pair equity curves (small multiples) ──────────────
    st.markdown(
        '<p style="font-family:monospace;font-size:10px;letter-spacing:0.12em;'
        'color:#4a5568;text-transform:uppercase;margin:0 0 12px;">Equity Curves by Pair</p>',
        unsafe_allow_html=True,
    )

    eq_cols = st.columns(len(all_pairs))
    for i, p in enumerate(all_pairs):
        r   = results[p["pair"]]
        eq  = r["equity_curve"]
        col = eq_cols[i]

        if len(eq) < 2:
            col.markdown(
                '<div style="background:#111318;border-radius:4px;'
                'border:1px solid rgba(255,255,255,0.07);padding:20px;text-align:center;">'
                '<p style="font-family:monospace;font-size:10px;color:#4a5568;">No trades</p></div>',
                unsafe_allow_html=True,
            )
            continue

        line_color = "#00d4a0" if eq.iloc[-1] >= BT_CAPITAL else "#f56565"
        fill_color = "rgba(0,212,160,0.07)" if eq.iloc[-1] >= BT_CAPITAL else "rgba(245,101,101,0.07)"

        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(
            x=eq.index, y=eq,
            fill="tozeroy", fillcolor=fill_color,
            line=dict(color=line_color, width=2),
        ))
        fig_eq.add_hline(
            y=BT_CAPITAL,
            line=dict(color="rgba(255,255,255,0.15)", width=1, dash="dot"),
        )
        fig_eq.update_layout(
            template="plotly_dark", height=140,
            paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
            margin=dict(l=4, r=4, t=4, b=4),
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, tickprefix="$", tickfont=dict(size=9)),
        )
        with col:
            st.markdown(
                '<p style="font-family:monospace;font-size:11px;color:#8892a4;'
                'margin:0 0 4px;text-align:center;">' + p["pair"] + "</p>",
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig_eq, use_container_width=True)


# =============================================================================
# 7. MAIN DASHBOARD
# =============================================================================
def main():
    st.title("📟 Omni-Arb Terminal v5.3")
    st.caption(
        f"Asset Universe: S&P 500 Pairs  |  Engine: Dynamic Cointegration  |  "
        f"Live Capital: ${STARTING_CAPITAL}  |  Backtest Capital: ${BT_CAPITAL}/pair"
    )
    st.divider()

    with st.spinner("Fetching market data..."):
        data_raw = get_market_data()

    all_pairs, active_pairs = process_pairs(data_raw)

    # ── Active signals ────────────────────────────────────────
    if active_pairs:
        st.subheader("⚠️ ACTIVE TRADE SIGNALS")
        for sig in active_pairs:
            render_trade_card(sig)
    else:
        st.info("✨ No active deviations detected. Monitoring market for entry...")

    st.divider()

    # ── Pair analysis charts ──────────────────────────────────
    st.subheader("📊 PAIR ANALYSIS")
    chart_cols = st.columns(2)

    for i, p in enumerate(all_pairs):
        with chart_cols[i % 2]:
            z_data = p["z_series"].dropna()
            fig = go.Figure()

            # Z-score line — thicker (width=2.5)
            fig.add_trace(go.Scatter(
                x=z_data.index, y=z_data,
                line=dict(color="#00d1ff", width=2.5),
                name="Spread Z", opacity=0.9,
            ))

            # Entry/exit markers — larger
            fig.add_trace(go.Scatter(
                x=p["long_spots"].index, y=p["long_spots"],
                mode="markers",
                marker=dict(color="#00ffcc", size=11, symbol="triangle-up"),
                name="ENTRY (BUY)",
            ))
            fig.add_trace(go.Scatter(
                x=p["short_spots"].index, y=p["short_spots"],
                mode="markers",
                marker=dict(color="#ff4b4b", size=11, symbol="triangle-down"),
                name="ENTRY (SELL)",
            ))
            fig.add_trace(go.Scatter(
                x=p["exit_spots"].index, y=p["exit_spots"],
                mode="markers",
                marker=dict(color="#ffffff", size=7, symbol="diamond"),
                name="EXIT",
            ))

            # Threshold lines — slightly thicker
            fig.add_hline(y=EXIT_Z,   line=dict(color="#ffffff",  width=1.2, dash="dot"),
                          annotation_text="EXIT (0.0)",         annotation_position="right")
            fig.add_hline(y=ENTRY_Z,  line=dict(color="#ff4b4b", width=1.5),
                          annotation_text="SELL ENTRY (+2.25)", annotation_position="top left")
            fig.add_hline(y=-ENTRY_Z, line=dict(color="#00ffcc", width=1.5),
                          annotation_text="BUY ENTRY (-2.25)",  annotation_position="bottom left")
            fig.add_hline(y=STOP_Z,   line=dict(color="#f5a623", width=2.2, dash="dash"),
                          annotation_text="STOP LOSS (+3.5)",   annotation_position="top right")
            fig.add_hline(y=-STOP_Z,  line=dict(color="#f5a623", width=2.2, dash="dash"),
                          annotation_text="STOP LOSS (-3.5)",   annotation_position="bottom right")

            last_date   = z_data.index[-1]
            x_range_end = last_date + (last_date - z_data.index[0]) * 0.15

            fig.update_layout(
                template="plotly_dark", height=380,
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
                showlegend=False,
                xaxis=dict(range=[z_data.index[0], x_range_end], showgrid=False),
                yaxis=dict(range=[-4, 4], showgrid=False, zeroline=False),
                font=dict(family="IBM Plex Mono"),
            )

            st.markdown(f"### {p['a']} vs {p['b']}")
            st.plotly_chart(fig, use_container_width=True)

            coint_color = "#00ffcc" if p["is_cointegrated"] else "#f5a623"
            coint_label = "HEALTHY" if p["is_cointegrated"] else "DRIFTING"
            st.markdown(
                '<div style="display:flex;justify-content:space-between;font-size:12px;'
                'color:#4a5568;margin-top:-15px;padding:0 10px;">'
                '<span>Beta: <b>' + f"{p['beta']:.2f}" + "</b></span>"
                '<span>Cointegration: <b style="color:' + coint_color + ';">' + coint_label + "</b></span>"
                "</div><br>",
                unsafe_allow_html=True,
            )

    # ── Backtest section ──────────────────────────────────────
    render_backtest_section(all_pairs)


if __name__ == "__main__":
    main()

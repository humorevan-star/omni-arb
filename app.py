# =============================================================================
# OMNI-ARB v5.4  |  Statistical Arbitrage Terminal
# State-machine backtest: Entry Hysteresis / Toggle Logic
# One-In One-Out per pair, shared $2,000 capital pool
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

# =============================================================================
# 0. CONFIG & THEME
# =============================================================================
st.set_page_config(page_title="Omni-Arb v5.4", layout="wide", initial_sidebar_state="collapsed")

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
PAIRS            = [('XOM', 'CVX'), ('V', 'MA'), ('NVDA', 'AMD'), ('KO', 'PEP'), ('GOOGL', 'META')]
ENTRY_Z          = 2.25
STOP_Z           = 3.5
EXIT_Z           = 0.0
ROLLING_WINDOW   = 60
STARTING_CAPITAL = 1000
BT_CAPITAL       = 2000   # shared pool across all pairs


# =============================================================================
# 2. SIZING  (live signal cards)
# =============================================================================
def compute_legs(sig, capital=STARTING_CAPITAL):
    target_per_leg = capital / 2
    shares_a = max(1, int(target_per_leg / sig["price_a"]))
    shares_b = max(1, int(shares_a * sig["beta"]))
    return {
        "shares_a":   shares_a,
        "notional_a": shares_a * sig["price_a"],
        "shares_b":   shares_b,
        "notional_b": shares_b * sig["price_b"],
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

        # ── Open trade detection (state-machine replay) ──────────────────
        # Walk the full z_series to find if the most recent entry cross
        # is still unclosed (no zero-cross or stop hit since then).
        in_open         = False
        open_direction  = None
        open_entry_date = None
        open_entry_z    = None

        for _i in range(1, len(z_series)):
            _zp = z_series.iloc[_i - 1]
            _zc = z_series.iloc[_i]
            _dt = z_series.index[_i]

            if not in_open:
                if _zp > -ENTRY_Z and _zc <= -ENTRY_Z:
                    in_open = True; open_direction = "LONG"
                    open_entry_date = _dt; open_entry_z = _zc
                elif _zp < ENTRY_Z and _zc >= ENTRY_Z:
                    in_open = True; open_direction = "SHORT"
                    open_entry_date = _dt; open_entry_z = _zc
            else:
                closed = (
                    (open_direction == "LONG"  and _zp < 0 and _zc >= 0) or
                    (open_direction == "SHORT" and _zp > 0 and _zc <= 0) or
                    abs(_zc) >= STOP_Z
                )
                if closed:
                    in_open = False; open_direction = None
                    open_entry_date = None; open_entry_z = None

        open_trade = (
            {"direction": open_direction, "entry_date": open_entry_date,
             "entry_z": open_entry_z, "curr_z": curr_z}
            if in_open else None
        )

        # Direction: at-threshold OR currently in an open trade
        direction_now = (
            "LONG"    if curr_z <= -ENTRY_Z else
            "SHORT"   if curr_z >= ENTRY_Z  else
            (open_direction if open_trade else "NEUTRAL")
        )

        info = {
            "pair":          f"{ticker_a}/{ticker_b}",
            "a":             ticker_a,
            "b":             ticker_b,
            "price_a":       pair_df[ticker_a].iloc[-1],
            "price_b":       pair_df[ticker_b].iloc[-1],
            "curr_z":        curr_z,
            "beta":          betas.iloc[-1],
            "z_series":      z_series,
            "spread":        spread,
            "pair_df":       pair_df,
            "betas_series":  betas,
            "consts_series": consts,
            "long_spots":    long_entries,
            "short_spots":   short_entries,
            "exit_spots":    exits,
            "is_cointegrated": adf_pval < 0.05,
            "adf_pval":      adf_pval,
            "open_trade":    open_trade,
            "direction":     direction_now,
        }
        processed.append(info)
        if info["direction"] != "NEUTRAL":
            active.append(info)
    return processed, active


# =============================================================================
# 4. STATE-MACHINE BACKTEST ENGINE
# =============================================================================
# Rules:
#   ONE position per pair at a time (toggle / entry hysteresis)
#   Entry fires ONLY on first cross of ENTRY_Z; all re-crosses ignored while in-trade
#   Exit fires on zero-cross (z -> 0.0) OR stop-loss hit (|z| >= STOP_Z)
#   Shared capital pool: each pair allocated BT_CAPITAL / n_pairs
#   If allocated capital <= 10, new entries skipped (waitlist)
# =============================================================================

def run_state_machine_backtest(all_pairs: list, total_capital: float = BT_CAPITAL) -> dict:
    n_pairs   = len(all_pairs)
    alloc_per = total_capital / n_pairs

    # Align each pair's data to a 2-year window
    z_map     = {}
    price_map = {}
    beta_map  = {}

    for p in all_pairs:
        z   = p["z_series"].dropna()
        pdf = p["pair_df"].reindex(z.index).dropna()
        bs  = p["betas_series"].reindex(z.index).ffill()
        common = z.index.intersection(pdf.index).intersection(bs.index)
        cutoff = common[-1] - pd.DateOffset(days=730)
        common = common[common >= cutoff]
        z_map[p["pair"]]     = z.loc[common]
        price_map[p["pair"]] = pdf.loc[common]
        beta_map[p["pair"]]  = bs.loc[common]

    # Union of all trading dates
    all_dates = sorted(set().union(*[s.index for s in z_map.values()]))

    # Per-pair state machine
    state = {
        p["pair"]: {
            "in_trade":   False,
            "direction":  None,
            "entry_date": None,
            "entry_pa":   None,
            "entry_pb":   None,
            "shares_a":   0,
            "shares_b":   0,
            "equity":     alloc_per,
            "trades":     [],
        }
        for p in all_pairs
    }

    portfolio_dates  = []
    portfolio_equity = []
    pair_eq_dates    = {p["pair"]: [] for p in all_pairs}
    pair_eq_values   = {p["pair"]: [] for p in all_pairs}

    for date in all_dates:
        for p in all_pairs:
            pk = p["pair"]
            ta = p["a"]
            tb = p["b"]
            zs = z_map[pk]
            ps = price_map[pk]
            bs = beta_map[pk]
            st_ = state[pk]

            if date not in zs.index:
                continue

            idx  = zs.index.get_loc(date)
            curr = zs.iloc[idx]
            prev = zs.iloc[idx - 1] if idx > 0 else curr
            pa   = ps[ta].loc[date]
            pb   = ps[tb].loc[date]
            bv   = bs.loc[date]

            if not st_["in_trade"]:
                long_cross  = prev > -ENTRY_Z and curr <= -ENTRY_Z
                short_cross = prev < ENTRY_Z  and curr >= ENTRY_Z

                if (long_cross or short_cross) and st_["equity"] > 10:
                    target   = st_["equity"] / 2
                    shares_a = max(1, int(target / pa))
                    shares_b = max(1, int(shares_a * abs(bv)))
                    st_["in_trade"]   = True
                    st_["direction"]  = "LONG" if long_cross else "SHORT"
                    st_["entry_date"] = date
                    st_["entry_pa"]   = pa
                    st_["entry_pb"]   = pb
                    st_["shares_a"]   = shares_a
                    st_["shares_b"]   = shares_b

            else:
                direction = st_["direction"]
                hit_exit  = (
                    (direction == "LONG"  and prev < 0 and curr >= 0) or
                    (direction == "SHORT" and prev > 0 and curr <= 0)
                )
                hit_stop = abs(curr) >= STOP_Z

                if hit_exit or hit_stop:
                    sa = st_["shares_a"]
                    sb = st_["shares_b"]
                    if direction == "LONG":
                        pnl = (sa * (pa - st_["entry_pa"])) - (sb * (pb - st_["entry_pb"]))
                    else:
                        pnl = (sa * (st_["entry_pa"] - pa)) - (sb * (st_["entry_pb"] - pb))

                    st_["equity"] += pnl
                    hold_days = (date - st_["entry_date"]).days

                    st_["trades"].append({
                        "pair":        pk,
                        "entry_date":  st_["entry_date"],
                        "exit_date":   date,
                        "direction":   direction,
                        "pnl":         pnl,
                        "pnl_pct":     pnl / alloc_per * 100,
                        "hold_days":   hold_days,
                        "exit_reason": "STOP" if hit_stop else "EXIT",
                        "entry_z":     zs.loc[st_["entry_date"]] if st_["entry_date"] in zs.index else float("nan"),
                        "exit_z":      curr,
                    })

                    st_["in_trade"]   = False
                    st_["direction"]  = None
                    st_["entry_date"] = None
                    st_["entry_pa"]   = None
                    st_["entry_pb"]   = None
                    st_["shares_a"]   = 0
                    st_["shares_b"]   = 0

            pair_eq_dates[pk].append(date)
            pair_eq_values[pk].append(st_["equity"])

        port_val = sum(s["equity"] for s in state.values())
        portfolio_dates.append(date)
        portfolio_equity.append(port_val)

    # Aggregate
    portfolio_series = pd.Series(portfolio_equity, index=portfolio_dates).groupby(level=0).last()

    all_trades     = []
    per_pair_stats = {}

    for p in all_pairs:
        pk     = p["pair"]
        trades = state[pk]["trades"]
        all_trades.extend(trades)

        eq = pd.Series(pair_eq_values[pk], index=pair_eq_dates[pk]).groupby(level=0).last()

        if not trades:
            per_pair_stats[pk] = {
                "num_trades": 0, "num_wins": 0, "num_losses": 0,
                "total_pnl": 0, "win_rate": 0, "apy": 0,
                "max_drawdown": 0, "avg_hold": 0,
                "best_trade": 0, "worst_trade": 0,
                "equity_curve": eq, "final_equity": alloc_per,
            }
            continue

        tdf       = pd.DataFrame(trades)
        total_pnl = tdf["pnl"].sum()
        num_wins  = int((tdf["pnl"] > 0).sum())
        num_loss  = int((tdf["pnl"] <= 0).sum())
        win_rate  = num_wins / len(tdf) * 100
        days_span = max((eq.index[-1] - eq.index[0]).days, 1)
        final_eq  = eq.iloc[-1]
        apy       = ((final_eq / alloc_per) ** (365 / days_span) - 1) * 100
        roll_max  = eq.cummax()
        max_dd    = ((eq - roll_max) / roll_max * 100).min()

        per_pair_stats[pk] = {
            "num_trades":   len(tdf),
            "num_wins":     num_wins,
            "num_losses":   num_loss,
            "total_pnl":    total_pnl,
            "win_rate":     win_rate,
            "apy":          apy,
            "max_drawdown": max_dd,
            "avg_hold":     tdf["hold_days"].mean(),
            "best_trade":   tdf["pnl"].max(),
            "worst_trade":  tdf["pnl"].min(),
            "equity_curve": eq,
            "final_equity": final_eq,
            "trade_df":     tdf,
        }

    total_trades = len(all_trades)
    if all_trades:
        all_tdf    = pd.DataFrame(all_trades)
        total_wins = int((all_tdf["pnl"] > 0).sum())
        total_loss = int((all_tdf["pnl"] <= 0).sum())
        total_pnl  = all_tdf["pnl"].sum()
        port_wr    = total_wins / total_trades * 100
        avg_hold   = all_tdf["hold_days"].mean()
    else:
        total_wins = total_loss = 0
        total_pnl  = port_wr = avg_hold = 0

    days_span = max((portfolio_series.index[-1] - portfolio_series.index[0]).days, 1)
    port_apy  = ((portfolio_series.iloc[-1] / total_capital) ** (365 / days_span) - 1) * 100
    roll_max  = portfolio_series.cummax()
    port_dd   = ((portfolio_series - roll_max) / roll_max * 100).min()

    return {
        "portfolio_equity": portfolio_series,
        "per_pair":         per_pair_stats,
        "all_trades":       all_trades,
        "total_trades":     total_trades,
        "total_wins":       total_wins,
        "total_losses":     total_loss,
        "total_pnl":        total_pnl,
        "port_wr":          port_wr,
        "port_apy":         port_apy,
        "port_dd":          port_dd,
        "avg_hold":         avg_hold,
        "alloc_per":        alloc_per,
    }


# =============================================================================
# 5. UI HELPERS
# =============================================================================
def render_trade_card(sig):
    is_long     = sig["direction"] == "LONG"
    accent      = "#00ffcc" if is_long else "#ff4b4b"
    bg          = "rgba(0, 255, 204, 0.05)" if is_long else "rgba(255, 75, 75, 0.05)"
    legs        = compute_legs(sig)
    leg1_verb   = "BUY"            if is_long else "SELL"
    leg2_verb   = "SELL"           if is_long else "BUY"
    leg1_col    = "#00ffcc"        if is_long else "#ff4b4b"
    leg2_col    = "#ff4b4b"        if is_long else "#00ffcc"
    action_text = "BUY THE SPREAD" if is_long else "SELL THE SPREAD"
    state_text  = "Lagging"        if is_long else "Overextended"
    z_str       = f"{sig['curr_z']:.2f}"

    st.markdown(
        '<div style="background:' + bg + ';border:1px solid ' + accent + ';padding:20px;'
        'border-radius:8px;margin-bottom:20px;">'
        '<div style="display:flex;justify-content:space-between;align-items:center;">'
        '<h2 style="margin:0;color:' + accent + ';font-size:24px;font-family:monospace;">'
        + sig["a"] + " / " + sig["b"] + "</h2>"
        '<div style="text-align:right;">'
        '<span style="font-size:12px;color:#8892a4;">CURRENT Z-SCORE</span><br>'
        '<b style="font-size:20px;color:' + accent + ';">' + z_str + "</b>"
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
# 6. BACKTEST DASHBOARD
# =============================================================================
def render_backtest_section(all_pairs: list):
    st.divider()
    st.markdown(
        '<p style="font-family:monospace;font-size:11px;letter-spacing:0.14em;'
        'color:#4a5568;text-transform:uppercase;margin-bottom:4px;">'
        '2-Year Historical Simulation  |  State-Machine  |  Entry Hysteresis / Toggle Logic</p>',
        unsafe_allow_html=True,
    )
    st.subheader("📈 Backtest Results  |  $2,000 Shared Capital Pool")

    with st.spinner("Running state-machine backtest..."):
        bt = run_state_machine_backtest(all_pairs, total_capital=BT_CAPITAL)

    port_eq       = bt["portfolio_equity"]
    total_capital = BT_CAPITAL

    pnl_color = "#00d4a0" if bt["total_pnl"] >= 0 else "#f56565"
    apy_color = "#00d4a0" if bt["port_apy"]  >= 0 else "#f56565"
    dd_color  = "#f5a623" if bt["port_dd"]   > -10 else "#f56565"

    # ── KPI strip ─────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)

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

    kpi(k1, "Total P&L",     f"${bt['total_pnl']:+,.0f}",                 pnl_color)
    kpi(k2, "Portfolio APY", f"{bt['port_apy']:+.1f}%",                    apy_color)
    kpi(k3, "Win Rate",      f"{bt['port_wr']:.0f}%",                     "#e8c96d")
    kpi(k4, "Total Trades",  str(bt["total_trades"]),                      "#e8eaf0")
    kpi(k5, "W / L",         f"{bt['total_wins']} / {bt['total_losses']}", "#e8eaf0")
    kpi(k6, "Max Drawdown",  f"{bt['port_dd']:.1f}%",                     dd_color)
    kpi(k7, "Avg Hold",      f"{bt['avg_hold']:.0f}d" if bt["avg_hold"] else "–", "#a78bfa")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SINGLE PORTFOLIO CHART ────────────────────────────────
    # Dim individual pair equity curves + bold portfolio total on top
    fig = go.Figure()

    pair_colors = ["#4a9eff", "#a78bfa", "#f5a623", "#e8c96d", "#f687b3"]

    for i, p in enumerate(all_pairs):
        ps = bt["per_pair"].get(p["pair"], {})
        eq = ps.get("equity_curve", pd.Series(dtype=float))
        if len(eq) < 2:
            continue
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq,
            mode="lines",
            line=dict(color=pair_colors[i % len(pair_colors)], width=1.2),
            opacity=0.4,
            name=p["pair"],
            hovertemplate="<b>" + p["pair"] + "</b><br>%{x|%b %d %Y}<br>$%{y:,.0f}<extra></extra>",
        ))

    # Bold portfolio total
    fig.add_trace(go.Scatter(
        x=port_eq.index, y=port_eq,
        mode="lines",
        fill="tozeroy",
        fillcolor="rgba(0,212,160,0.06)",
        line=dict(color="#00d4a0", width=3),
        name="Portfolio Total",
        hovertemplate="<b>Portfolio</b><br>%{x|%b %d %Y}<br>$%{y:,.0f}<extra></extra>",
    ))

    # Starting capital reference
    fig.add_hline(
        y=total_capital,
        line=dict(color="rgba(255,255,255,0.18)", width=1, dash="dot"),
        annotation_text="Starting Capital  $" + f"{total_capital:,}",
        annotation_font_color="#4a5568",
        annotation_font_size=10,
        annotation_position="top left",
    )

    # Entry / exit markers mapped onto portfolio equity curve
    if bt["all_trades"]:
        all_tdf = pd.DataFrame(bt["all_trades"])

        entry_dates  = all_tdf["entry_date"].tolist()
        entry_values = [port_eq.asof(d) if d >= port_eq.index[0] else None for d in entry_dates]
        entry_colors = ["#00d4a0" if d == "LONG" else "#f56565" for d in all_tdf["direction"].tolist()]

        fig.add_trace(go.Scatter(
            x=entry_dates, y=entry_values,
            mode="markers",
            marker=dict(color=entry_colors, size=9, symbol="circle",
                        line=dict(color="#0b0e14", width=1.5)),
            name="Entry",
            hovertemplate="<b>ENTRY</b>  %{x|%b %d %Y}<extra></extra>",
        ))

        exit_dates  = all_tdf["exit_date"].tolist()
        exit_values = [port_eq.asof(d) if d >= port_eq.index[0] else None for d in exit_dates]
        exit_colors = ["#f5a623" if r == "STOP" else "#ffffff" for r in all_tdf["exit_reason"].tolist()]

        fig.add_trace(go.Scatter(
            x=exit_dates, y=exit_values,
            mode="markers",
            marker=dict(color=exit_colors, size=8, symbol="diamond",
                        line=dict(color="#0b0e14", width=1.5)),
            name="Exit",
            hovertemplate="<b>EXIT</b>  %{x|%b %d %Y}<extra></extra>",
        ))

    y_lo = min(port_eq.min() * 0.97, total_capital * 0.93) if len(port_eq) else 0
    y_hi = max(port_eq.max() * 1.03, total_capital * 1.05) if len(port_eq) else total_capital * 1.1

    fig.update_layout(
        template="plotly_dark",
        height=420,
        paper_bgcolor="#0b0e14",
        plot_bgcolor="#0b0e14",
        margin=dict(l=12, r=12, t=12, b=12),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
            font=dict(family="IBM Plex Mono", size=10, color="#8892a4"),
            bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(
            showgrid=False,
            rangeselector=dict(
                bgcolor="#111318", activecolor="#00d4a0",
                bordercolor="rgba(255,255,255,0.1)",
                font=dict(family="IBM Plex Mono", size=10, color="#8892a4"),
                buttons=[
                    dict(count=3,  label="3M", step="month", stepmode="backward"),
                    dict(count=6,  label="6M", step="month", stepmode="backward"),
                    dict(count=1,  label="1Y", step="year",  stepmode="backward"),
                    dict(step="all", label="All"),
                ],
            ),
        ),
        yaxis=dict(showgrid=False, zeroline=False, tickprefix="$", range=[y_lo, y_hi]),
        hovermode="x unified",
        font=dict(family="IBM Plex Mono"),
    )

    st.plotly_chart(fig, use_container_width=True)

    # Legend explainer
    st.markdown(
        '<div style="display:flex;gap:24px;padding:0 4px 16px;'
        'font-family:monospace;font-size:11px;color:#4a5568;">'
        '<span><span style="color:#00d4a0;">●</span> LONG entry</span>'
        '<span><span style="color:#f56565;">●</span> SHORT entry</span>'
        '<span><span style="color:#ffffff;">◆</span> Exit (zero-cross)</span>'
        '<span><span style="color:#f5a623;">◆</span> Exit (stop-loss)</span>'
        '<span>── Pair curves</span>'
        '<span style="color:#00d4a0;font-weight:600;">─── Portfolio total</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    # ── Per-pair performance cards ────────────────────────────
    st.markdown(
        '<p style="font-family:monospace;font-size:10px;letter-spacing:0.12em;'
        'color:#4a5568;text-transform:uppercase;margin:4px 0 14px;">Per-Pair Performance</p>',
        unsafe_allow_html=True,
    )

    def row(label, value, val_color="#e8eaf0"):
        return (
            '<div style="display:flex;justify-content:space-between;margin-bottom:5px;">'
            '<span style="font-size:10px;color:#4a5568;font-family:monospace;">' + label + "</span>"
            '<span style="font-family:monospace;font-size:12px;color:' + val_color + ';">' + value + "</span>"
            "</div>"
        )

    pair_cols = st.columns(len(all_pairs))
    for i, p in enumerate(all_pairs):
        pk  = p["pair"]
        r   = bt["per_pair"].get(pk, {})
        col = pair_cols[i]

        pnl_c   = "#00d4a0" if r.get("total_pnl", 0) >= 0 else "#f56565"
        apy_c   = "#00d4a0" if r.get("apy", 0)       >= 0 else "#f56565"
        coint_c = "#00d4a0" if p["is_cointegrated"]        else "#f5a623"

        alloc_v   = f"${bt['alloc_per']:,.0f}"
        final_v   = f"${r.get('final_equity', bt['alloc_per']):,.0f}"
        pnl_v     = f"${r.get('total_pnl', 0):+,.0f}"
        apy_v     = f"{r.get('apy', 0):+.1f}%"
        wr_v      = f"{r.get('win_rate', 0):.0f}%"
        trades_v  = str(r.get("num_trades", 0))
        wl_v      = f"{r.get('num_wins', 0)}W / {r.get('num_losses', 0)}L"
        hold_v    = f"{r.get('avg_hold', 0):.0f}d" if r.get("num_trades") else "–"
        dd_v      = f"{r.get('max_drawdown', 0):.1f}%"
        best_v    = f"${r.get('best_trade', 0):+,.0f}" if r.get("num_trades") else "–"
        worst_v   = f"${r.get('worst_trade', 0):+,.0f}" if r.get("num_trades") else "–"
        c_label   = "COINTEGRATED" if p["is_cointegrated"] else "DRIFTING"

        col.markdown(
            '<div style="background:#111318;padding:14px;border-radius:4px;'
            'border:1px solid rgba(255,255,255,0.07);">'
            '<p style="margin:0 0 10px;font-family:monospace;font-size:13px;'
            'font-weight:600;color:#e8eaf0;">' + pk + "</p>"
            + row("Alloc",    alloc_v)
            + row("Final",    final_v,  pnl_c)
            + row("P&amp;L",  pnl_v,    pnl_c)
            + row("APY",      apy_v,    apy_c)
            + row("Win Rate", wr_v,     "#e8c96d")
            + row("Trades",   trades_v)
            + row("W / L",    wl_v)
            + row("Avg Hold", hold_v,   "#a78bfa")
            + row("Max DD",   dd_v,     "#f5a623")
            + '<div style="border-top:1px solid rgba(255,255,255,0.07);margin:8px 0;">'
            + row("Best",  best_v,  "#00d4a0")
            + row("Worst", worst_v, "#f56565")
            + "</div>"
            '<div style="margin-top:8px;text-align:center;">'
            '<span style="font-size:9px;font-family:monospace;padding:2px 8px;border-radius:2px;'
            'color:' + coint_c + ';border:1px solid ' + coint_c + ';opacity:0.8;">'
            + c_label + "</span></div>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── State-machine explainer ───────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="background:#111318;border:1px solid rgba(255,255,255,0.07);'
        'border-left:3px solid #4a9eff;border-radius:4px;padding:16px 20px;">'
        '<p style="margin:0 0 10px;font-family:monospace;font-size:10px;color:#4a9eff;'
        'text-transform:uppercase;letter-spacing:0.1em;">State-Machine Logic — Entry Hysteresis / Toggle</p>'
        '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:16px;">'
        '<div><p style="margin:0 0 4px;font-size:11px;color:#e8eaf0;font-family:monospace;">① Entry</p>'
        '<p style="margin:0;font-size:10px;color:#4a5568;line-height:1.6;">'
        'Z crosses ±2.25. Position opens. Switch flips ON. '
        'All subsequent re-crosses are ignored — no double-down, no averaging-in.</p></div>'
        '<div><p style="margin:0 0 4px;font-size:11px;color:#e8eaf0;font-family:monospace;">② Noise Phase</p>'
        '<p style="margin:0;font-size:10px;color:#4a5568;line-height:1.6;">'
        'Z wiggles above/below the threshold. System does nothing. '
        'Capital is protected. Opportunity cost: your $400 is locked until exit.</p></div>'
        '<div><p style="margin:0 0 4px;font-size:11px;color:#e8eaf0;font-family:monospace;">③ Exit &amp; Reset</p>'
        '<p style="margin:0;font-size:10px;color:#4a5568;line-height:1.6;">'
        'Z returns to 0.0 → both legs close, P&L realised, switch resets to NEUTRAL. '
        'Stop-loss at ±3.5 cuts losses early if spread diverges further.</p></div>'
        "</div></div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# 7. MAIN DASHBOARD
# =============================================================================
def main():
    st.title("📟 Omni-Arb Terminal v5.4")
    st.caption(
        f"Asset Universe: S&P 500 Pairs  |  Engine: State-Machine Cointegration  |  "
        f"Live Capital: ${STARTING_CAPITAL}  |  Backtest Pool: ${BT_CAPITAL} shared"
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
            z_data     = p["z_series"].dropna()
            open_trade = p.get("open_trade")
            is_long    = (open_trade or {}).get("direction") == "LONG"
            is_short   = (open_trade or {}).get("direction") == "SHORT"

            fig = go.Figure()

            # ── Active trade zone shading ─────────────────────
            # Shades from entry_z toward 0.0 to show remaining profit room
            if open_trade:
                entry_z_val  = open_trade["entry_z"]
                shade_y_lo   = min(entry_z_val, EXIT_Z) if is_long  else EXIT_Z
                shade_y_hi   = EXIT_Z                   if is_long  else max(entry_z_val, EXIT_Z)
                if is_short:
                    shade_y_lo = EXIT_Z
                    shade_y_hi = entry_z_val
                shade_color  = "rgba(0,255,204,0.09)"   if is_long else "rgba(255,75,75,0.09)"
                shade_line   = "rgba(0,255,204,0.0)"    if is_long else "rgba(255,75,75,0.0)"

                # Filled band: phantom upper line + lower line with fill between
                fig.add_trace(go.Scatter(
                    x=list(z_data.index) + list(z_data.index[::-1]),
                    y=[shade_y_hi] * len(z_data) + [shade_y_lo] * len(z_data),
                    fill="toself",
                    fillcolor=shade_color,
                    line=dict(color="rgba(0,0,0,0)", width=0),
                    showlegend=False,
                    hoverinfo="skip",
                    name="Trade Zone",
                ))

            # ── Z-score line ──────────────────────────────────
            fig.add_trace(go.Scatter(
                x=z_data.index, y=z_data,
                line=dict(color="#00d1ff", width=2.5),
                name="Spread Z", opacity=0.9,
            ))

            # ── Historical entry/exit markers (dim, past trades) ─
            # Exclude the active entry marker — we'll draw it glowing separately
            active_entry_date = open_trade["entry_date"] if open_trade else None

            hist_long  = p["long_spots"]
            hist_short = p["short_spots"]
            if active_entry_date is not None:
                hist_long  = hist_long[hist_long.index  != active_entry_date]
                hist_short = hist_short[hist_short.index != active_entry_date]

            fig.add_trace(go.Scatter(
                x=hist_long.index, y=hist_long,
                mode="markers",
                marker=dict(color="#00ffcc", size=10, symbol="triangle-up", opacity=0.55),
                name="Entry (BUY)",
            ))
            fig.add_trace(go.Scatter(
                x=hist_short.index, y=hist_short,
                mode="markers",
                marker=dict(color="#ff4b4b", size=10, symbol="triangle-down", opacity=0.55),
                name="Entry (SELL)",
            ))
            fig.add_trace(go.Scatter(
                x=p["exit_spots"].index, y=p["exit_spots"],
                mode="markers",
                marker=dict(color="#ffffff", size=7, symbol="diamond", opacity=0.5),
                name="EXIT",
            ))

            # ── Glowing active entry marker ───────────────────
            if open_trade and active_entry_date in z_data.index:
                glow_color = "#00ffcc" if is_long else "#ff4b4b"
                glow_sym   = "triangle-up" if is_long else "triangle-down"
                entry_y    = open_trade["entry_z"]

                # Outer glow ring
                fig.add_trace(go.Scatter(
                    x=[active_entry_date], y=[entry_y],
                    mode="markers",
                    marker=dict(color="rgba(0,255,204,0.18)" if is_long else "rgba(255,75,75,0.18)",
                                size=28, symbol="circle"),
                    showlegend=False, hoverinfo="skip",
                ))
                # Middle ring
                fig.add_trace(go.Scatter(
                    x=[active_entry_date], y=[entry_y],
                    mode="markers",
                    marker=dict(color="rgba(0,255,204,0.35)" if is_long else "rgba(255,75,75,0.35)",
                                size=20, symbol="circle"),
                    showlegend=False, hoverinfo="skip",
                ))
                # Solid entry dot
                fig.add_trace(go.Scatter(
                    x=[active_entry_date], y=[entry_y],
                    mode="markers",
                    marker=dict(color=glow_color, size=13, symbol=glow_sym,
                                line=dict(color="#0b0e14", width=2)),
                    name="ACTIVE ENTRY",
                    hovertemplate=(
                        "<b>ACTIVE ENTRY (" + open_trade["direction"] + ")</b><br>"
                        + active_entry_date.strftime("%b %d %Y")
                        + "<br>Z at entry: " + f"{entry_y:.2f}"
                        + "<br>Z now: " + f"{p['curr_z']:.2f}"
                        + "<extra></extra>"
                    ),
                ))

                # Vertical dashed line from entry date to top of chart
                entry_date_str = active_entry_date.isoformat()
                fig.add_vline(
                    x=entry_date_str,
                    line=dict(color=glow_color, width=1.2, dash="dot"),
                )

            # ── Threshold lines ───────────────────────────────
            fig.add_hline(y=EXIT_Z,   line=dict(color="#ffffff",  width=1.2, dash="dot"),
                          annotation_text="EXIT (0.0)", annotation_position="right")
            fig.add_hline(y=ENTRY_Z,  line=dict(color="#ff4b4b", width=1.5),
                          annotation_text="SELL ENTRY (+2.25)", annotation_position="top left")
            fig.add_hline(y=-ENTRY_Z, line=dict(color="#00ffcc", width=1.5),
                          annotation_text="BUY ENTRY (-2.25)",  annotation_position="bottom left")
            fig.add_hline(y=STOP_Z,   line=dict(color="#f5a623", width=2.2, dash="dash"),
                          annotation_text="STOP LOSS (+3.5)",   annotation_position="top right")
            fig.add_hline(y=-STOP_Z,  line=dict(color="#f5a623", width=2.2, dash="dash"),
                          annotation_text="STOP LOSS (-3.5)",   annotation_position="bottom right")

            # ── Axis ranges ───────────────────────────────────
            last_date    = z_data.index[-1]
            one_year_ago = last_date - pd.DateOffset(years=1)
            x_start      = max(one_year_ago, z_data.index[0])
            x_pad        = (last_date - x_start) * 0.12   # wider right padding for status label
            x_end        = last_date + x_pad
            visible_z    = z_data[z_data.index >= x_start]
            y_min        = visible_z.min()
            y_max        = visible_z.max()
            y_pad        = (y_max - y_min) * 0.15 if (y_max - y_min) > 0 else 0.5
            y_lo         = min(y_min - y_pad, -STOP_Z - 0.3)
            y_hi         = max(y_max + y_pad,  STOP_Z + 0.3)

            # ── Floating STATUS label in right breathing room ─
            annotations = []
            if open_trade:
                status_color = "#00ffcc" if is_long else "#ff4b4b"
                status_dir   = open_trade["direction"]
                status_text  = "STATUS: OPEN (" + status_dir + ")"
                z_now_str    = f"Z now: {p['curr_z']:.2f}"

                # Label anchored to right edge at current z value
                annotations.append(dict(
                    x=x_end, y=p["curr_z"],
                    xref="x", yref="y",
                    text="<b>" + status_text + "</b><br><span style=\'font-size:10px\'>" + z_now_str + "</span>",
                    showarrow=True,
                    arrowhead=2, arrowsize=0.8,
                    arrowcolor=status_color,
                    ax=-80, ay=0,
                    font=dict(family="IBM Plex Mono", size=10, color=status_color),
                    bgcolor="rgba(11,14,20,0.85)",
                    bordercolor=status_color,
                    borderwidth=1,
                    borderpad=5,
                    align="left",
                ))
            else:
                annotations.append(dict(
                    x=x_end, y=p["curr_z"],
                    xref="x", yref="y",
                    text="<b>STATUS: NEUTRAL</b>",
                    showarrow=False,
                    font=dict(family="IBM Plex Mono", size=10, color="#4a5568"),
                    bgcolor="rgba(11,14,20,0.85)",
                    bordercolor="#1e2330",
                    borderwidth=1,
                    borderpad=5,
                ))

            fig.update_layout(
                template="plotly_dark", height=400,
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
                showlegend=False,
                annotations=annotations,
                xaxis=dict(
                    range=[x_start, x_end],
                    showgrid=False,
                    rangeslider=dict(visible=False),
                    rangeselector=dict(
                        bgcolor="#111318", activecolor="#00d1ff",
                        bordercolor="rgba(255,255,255,0.1)",
                        font=dict(family="IBM Plex Mono", size=10, color="#8892a4"),
                        buttons=[
                            dict(count=3,  label="3M", step="month", stepmode="backward"),
                            dict(count=6,  label="6M", step="month", stepmode="backward"),
                            dict(count=1,  label="1Y", step="year",  stepmode="backward"),
                            dict(step="all", label="All"),
                        ],
                    ),
                ),
                yaxis=dict(range=[y_lo, y_hi], showgrid=False, zeroline=False, autorange=False),
                font=dict(family="IBM Plex Mono"),
            )

            beta_str    = f"{p['beta']:.2f}"
            coint_color = "#00ffcc" if p["is_cointegrated"] else "#f5a623"
            coint_label = "HEALTHY"  if p["is_cointegrated"] else "DRIFTING"

            st.markdown(f"### {p['a']} vs {p['b']}")
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                '<div style="display:flex;justify-content:space-between;font-size:12px;'
                'color:#4a5568;margin-top:-15px;padding:0 10px;">'
                '<span>Beta: <b>' + beta_str + "</b></span>"
                '<span>Cointegration: <b style="color:' + coint_color + ';">' + coint_label + "</b></span>"
                "</div><br>",
                unsafe_allow_html=True,
            )

            # ── Options spread panel (shown only when trade is open) ──────
            if open_trade:
                _is_long   = open_trade["direction"] == "LONG"
                _dir       = open_trade["direction"]
                _entry_z   = open_trade["entry_z"]
                _curr_z    = p["curr_z"]
                _pa        = p["price_a"]
                _pb        = p["price_b"]
                _beta      = p["beta"]
                _a         = p["a"]
                _b         = p["b"]

                # Rough price-per-1-z-unit on leg A (used for strike approximation)
                # Strike at current price (entry) and fair-value price (exit = 0.0)
                # We estimate fair-value price as current price adjusted by spread move
                _spread_move   = abs(_entry_z - 0.0)   # z units to exit
                _pct_move_est  = _spread_move * 0.01    # ~1% per z unit (rough)
                _fair_pa       = round(_pa * (1 + _pct_move_est) if _is_long else _pa * (1 - _pct_move_est), 2)

                _spread_type   = "Bull Call Spread" if _is_long else "Bear Put Spread"
                _opt_type      = "Call"             if _is_long else "Put"
                _buy_leg       = "Buy  " + _opt_type + " @ $" + f"{_pa:.2f}"  + " (current price — cheap)"
                _sell_leg      = "Sell " + _opt_type + " @ $" + f"{_fair_pa:.2f}" + " (fair-value target — exit zone)"
                _max_risk_note = "Max risk = premium paid (defined by Stop @ ±" + str(STOP_Z) + ")"
                _max_rwd_note  = "Max reward = spread width if price reaches fair value"

                _accent    = "#00ffcc" if _is_long else "#ff4b4b"
                _bg        = "rgba(0,255,204,0.04)" if _is_long else "rgba(255,75,75,0.04)"
                _border    = "rgba(0,255,204,0.2)"  if _is_long else "rgba(255,75,75,0.2)"

                st.markdown(
                    '<div style="background:' + _bg + ';border:1px solid ' + _border + ';'
                    'border-radius:6px;padding:14px 18px;margin-bottom:20px;">'
                    '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">'
                    '<p style="margin:0;font-family:monospace;font-size:11px;font-weight:600;'
                    'text-transform:uppercase;letter-spacing:0.1em;color:' + _accent + ';'
                    '">' + _spread_type + " — " + _a + " (capital-efficient equivalent)</p>"
                    '<span style="font-family:monospace;font-size:9px;padding:2px 7px;border-radius:2px;'
                    'color:' + _accent + ';border:1px solid ' + _accent + ';opacity:0.7;">'
                    'OPEN — ' + _dir + "</span>"
                    "</div>"
                    '<div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;">'
                    "<div>"
                    '<p style="margin:0 0 6px;font-size:10px;color:#4a5568;font-family:monospace;'
                    'text-transform:uppercase;letter-spacing:0.08em;">Leg Structure</p>'
                    '<p style="margin:0 0 4px;font-family:monospace;font-size:12px;color:' + _accent + ';'
                    '">' + _buy_leg + "</p>"
                    '<p style="margin:0;font-family:monospace;font-size:12px;color:#8892a4;">'
                    + _sell_leg + "</p>"
                    "</div>"
                    "<div>"
                    '<p style="margin:0 0 6px;font-size:10px;color:#4a5568;font-family:monospace;'
                    'text-transform:uppercase;letter-spacing:0.08em;">Risk Profile</p>'
                    '<p style="margin:0 0 4px;font-family:monospace;font-size:11px;color:#e8c96d;">'
                    + _max_rwd_note + "</p>"
                    '<p style="margin:0;font-family:monospace;font-size:11px;color:#f5a623;">'
                    + _max_risk_note + "</p>"
                    "</div>"
                    "</div>"
                    "</div>",
                    unsafe_allow_html=True,
                )

    # ── Backtest section ──────────────────────────────────────
    render_backtest_section(all_pairs)


if __name__ == "__main__":
    main()

# =============================================================================
# OMNI-ARB v6.0  |  Statistical Arbitrage Terminal
# State-machine logic: Entry Hysteresis / Toggle Logic
# Focus: Execution-Centric Signals (Stocks & Options)
# Enhanced: Rich UX, clear explanations, active-trade zone charts
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
BT_CAPITAL       = 2000    # total pool
BT_MAX_OPEN      = 3       # max simultaneous open trades
BT_MAX_HOLD_DAYS = 21      # max calendar days held (~21 trading days)
BT_LOOKBACK_DAYS = 730     # 2-year window
BT_ALLOC_PER     = BT_CAPITAL / len(PAIRS)  # per-pair slice


# =============================================================================
# 2. SIZING
# =============================================================================
def compute_legs(sig, capital=STARTING_CAPITAL):
    """
    Medallion Delta-Neutral Sizing — fixes the "Price-Beta Trap".

    The naive formula (qty_b = qty_a * beta) confuses share counts with
    dollar exposure. If GOOGL=$175 and META=$580, the same number of shares
    represents very different dollar risk.

    Correct formula:
      dollar_a = capital / (1 + beta)          <- beta-weighted dollar split
      dollar_b = capital - dollar_a
      shares_a = dollar_a / price_a
      shares_b = dollar_b / price_b

    This guarantees:  dollar_b = beta * dollar_a
    So a 1-unit Z-score move produces the same P&L on both legs.

    Risk imbalance check:
      risk_imbalance = val_a - (val_b / beta)  <- should be near $0
    """
    price_a = float(sig["price_a"])
    price_b = float(sig["price_b"])
    beta    = max(abs(float(sig["beta"])), 0.01)   # floor at 0.01 — avoid /0

    # Step 1 — risk-neutral dollar allocation
    dollar_a = capital / (1.0 + beta)
    dollar_b = capital - dollar_a

    # Step 2 — convert dollars to shares at 0.1 precision
    shares_a = max(0.1, round(dollar_a / price_a, 1))
    shares_b = max(0.1, round(dollar_b / price_b, 1))

    # Step 3 — actual notionals after rounding
    notional_a = round(shares_a * price_a, 2)
    notional_b = round(shares_b * price_b, 2)
    total_cost = round(notional_a + notional_b, 2)

    # Risk imbalance: val_a - (val_b / beta) should be ~$0 if perfectly neutral
    risk_imbalance = round(notional_a - (notional_b / beta), 2)
    # Dollar imbalance (absolute, for colour threshold in card)
    dollar_imbal   = round(abs(notional_a - notional_b), 2)
    # Signed dollar imbalance for display
    imbalance      = round(notional_a - notional_b, 2)
    hedge_ratio    = round(shares_a / shares_b, 2) if shares_b != 0 else 0
    ratio_str      = str(shares_a) + ":" + str(shares_b)

    # Step 4 — spread P&L using entry-LOCKED shares (Medallion method)
    # Use locked shares from entry so sizing drift does not corrupt P&L.
    # Spread P&L = log(cp_a/ep_a) * notional_a  -  beta * log(cp_b/ep_b) * notional_b
    # This correctly measures whether the RELATIVE move (not absolute) favours us.
    pnl = pnl_a = pnl_b = None
    entry_pa = entry_pb = None
    pnl_potential = None   # remaining profit if trade closes at 0.0
    ot = sig.get("open_trade")
    if ot and ot.get("entry_price_a") and ot.get("entry_price_b"):
        import math as _math
        ep_a     = float(ot["entry_price_a"])
        ep_b     = float(ot["entry_price_b"])
        entry_pa = round(ep_a, 2)
        entry_pb = round(ep_b, 2)
        is_long  = ot["direction"] == "LONG"
        # Use entry-locked shares if available, else fall back to current sizing
        _sa = float(ot["entry_shares_a"]) if ot.get("entry_shares_a") else shares_a
        _sb = float(ot["entry_shares_b"]) if ot.get("entry_shares_b") else shares_b
        # Dollar P&L per leg (simple, for leg breakdown display)
        if is_long:
            pnl_a = round((price_a - ep_a) * _sa, 2)
            pnl_b = round(-(price_b - ep_b) * _sb, 2)
        else:
            pnl_a = round(-(price_a - ep_a) * _sa, 2)
            pnl_b = round((price_b - ep_b) * _sb, 2)
        pnl = round(pnl_a + pnl_b, 2)
        # Spread P&L = log-return of spread × notional_a
        # Positive when spread reverts toward 0 (regardless of market direction)
        notional_a_entry = _sa * ep_a
        log_spread_entry = _math.log(ep_a) - beta * _math.log(ep_b)
        log_spread_now   = _math.log(price_a) - beta * _math.log(price_b)
        delta_spread     = log_spread_now - log_spread_entry
        spread_pnl = round(
            (delta_spread * notional_a_entry) * (1 if is_long else -1), 2
        )
        # Remaining profit potential (if spread closes all the way to 0)
        entry_z_val  = float(ot.get("entry_z") or 0)
        curr_z_val   = float(sig["curr_z"])
        if entry_z_val != 0:
            pnl_per_z_unit = spread_pnl / max(abs(abs(entry_z_val) - abs(curr_z_val)), 0.01)
            pnl_potential  = round(pnl_per_z_unit * abs(curr_z_val), 2)
        pnl = spread_pnl   # override with spread-correct P&L

    return {
        # Core sizing
        "shares_a":       shares_a,
        "shares_b":       shares_b,
        "ratio":          ratio_str,
        # Notionals
        "notional_a":     notional_a,
        "notional_b":     notional_b,
        "total_cost":     total_cost,      # backward compat alias
        "total_deployed": total_cost,
        # Neutrality metrics
        "risk_imbalance": risk_imbalance,  # ~$0 = perfect delta-neutral
        "imbalance":      imbalance,       # signed dollar imbalance
        "dollar_imbal":   dollar_imbal,    # abs, for card colour threshold
        "hedge_ratio":    hedge_ratio,
        # P&L
        "pnl":            pnl,
        "pnl_a":          pnl_a,
        "pnl_b":          pnl_b,
        "pnl_potential":  pnl_potential,   # remaining profit if closed at 0
        # Entry prices for P&L calculation
        "entry_pa":       entry_pa,
        "entry_pb":       entry_pb,
    }


# =============================================================================
# 2b. REBALANCE ENGINE
# =============================================================================
def get_rebalance_instructions(sig: dict) -> dict:
    """
    48-hour beta-drift rebalance check (Medallion method).

    Every 2 calendar days, compares the ideal Medallion share count for
    leg B (recalculated using the CURRENT beta) against the entry-locked
    share count. If the drift is ≥ 0.1 shares, it issues a fractional
    adjustment instruction for leg B only.

    Why only leg B?
      Leg A sizing (shares_a) is determined by capital / (1+beta) and
      price_a. We keep leg A fixed to avoid unnecessary round-trips.
      Leg B is the hedge — its ratio drifts as beta evolves, so it is
      the only leg that ever needs a fractional tweak.

    Returns a dict with:
      status:   "STABLE" | "REBALANCE"
      action:   "BUY" | "SELL" | None
      ticker:   ticker_b
      qty:      fractional share adjustment (0.1 precision)
      reason:   human-readable explanation
      days_in:  calendar days since entry
    """
    ot = sig.get("open_trade")
    if not ot or not ot.get("entry_date"):
        return {"status": "STABLE", "action": None, "qty": 0,
                "ticker": sig["b"], "reason": "No open trade", "days_in": 0}

    try:
        days_in = (pd.Timestamp.now().normalize() - pd.Timestamp(ot["entry_date"]).normalize()).days
    except Exception:
        days_in = 0

    # Only check on even calendar days (≈ every 48 hours)
    if days_in <= 0 or days_in % 2 != 0:
        return {"status": "STABLE", "action": None, "qty": 0,
                "ticker": sig["b"],
                "reason": f"Day {days_in} — next check on day {days_in + (2 - days_in % 2) if days_in % 2 != 0 else days_in + 2}",
                "days_in": days_in}

    # Ideal share count for leg B using CURRENT beta + CURRENT prices
    current_beta  = max(abs(float(sig["beta"])), 0.01)
    d_a_ideal     = STARTING_CAPITAL / (1.0 + current_beta)
    d_b_ideal     = STARTING_CAPITAL - d_a_ideal
    ideal_sb      = max(0.1, round(d_b_ideal / float(sig["price_b"]), 1))

    # Compare against entry-locked shares
    locked_sb     = float(ot.get("entry_shares_b") or 0)
    diff_b        = round(ideal_sb - locked_sb, 1)

    if abs(diff_b) < 0.1:
        return {"status": "STABLE", "action": None, "qty": 0,
                "ticker": sig["b"],
                "reason": f"Beta drift negligible (Δ={diff_b:+.1f} shs). Hedge stable.",
                "days_in": days_in}

    action     = "BUY" if diff_b > 0 else "SELL"
    abs_diff   = abs(diff_b)
    notional   = round(abs_diff * float(sig["price_b"]), 2)
    reason     = (
        f"β drifted → ideal {sig['b']} = {ideal_sb} shs, locked = {locked_sb} shs. "
        f"Adjust by {diff_b:+.1f} shs (${notional:,.2f}) to restore delta-neutrality."
    )

    return {
        "status":    "REBALANCE",
        "action":    action,
        "ticker":    sig["b"],
        "qty":       abs_diff,
        "diff":      diff_b,
        "ideal_sb":  ideal_sb,
        "locked_sb": locked_sb,
        "notional":  notional,
        "reason":    reason,
        "days_in":   days_in,
    }


# =============================================================================
# 3. DATA ENGINE  (State-Machine Processor)
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
        ).dropna()

        curr_z   = z_series.iloc[-1]
        adf_pval = adfuller(spread)[1]

        in_open          = False
        open_direction   = None
        open_entry_z     = None
        open_entry_date  = None
        open_entry_pa    = None   # price of leg A at entry
        open_entry_pb    = None   # price of leg B at entry

        for _i in range(1, len(z_series)):
            _zp = z_series.iloc[_i - 1]
            _zc = z_series.iloc[_i]
            _dt = z_series.index[_i]

            if not in_open:
                if _zp > -ENTRY_Z and _zc <= -ENTRY_Z:
                    in_open          = True
                    open_direction   = "LONG"
                    open_entry_z     = _zc
                    open_entry_date  = _dt
                    open_entry_pa    = pair_df[ticker_a].loc[_dt] if _dt in pair_df.index else None
                    open_entry_pb    = pair_df[ticker_b].loc[_dt] if _dt in pair_df.index else None
                elif _zp < ENTRY_Z and _zc >= ENTRY_Z:
                    in_open          = True
                    open_direction   = "SHORT"
                    open_entry_z     = _zc
                    open_entry_date  = _dt
                    open_entry_pa    = pair_df[ticker_a].loc[_dt] if _dt in pair_df.index else None
                    open_entry_pb    = pair_df[ticker_b].loc[_dt] if _dt in pair_df.index else None
            else:
                closed = (
                    (open_direction == "LONG"  and _zp < 0 and _zc >= 0) or
                    (open_direction == "SHORT" and _zp > 0 and _zc <= 0) or
                    abs(_zc) >= STOP_Z
                )
                if closed:
                    in_open          = False
                    open_direction   = None
                    open_entry_z     = None
                    open_entry_date  = None
                    open_entry_pa    = None
                    open_entry_pb    = None

        pct_to_target = 0.0
        if in_open and open_entry_z and open_entry_z != 0:
            pct_to_target = round(
                (abs(open_entry_z) - abs(curr_z)) / abs(open_entry_z) * 100, 1
            )

        # Compute entry-locked shares using Medallion formula at entry prices
        if in_open and open_entry_pa and open_entry_pb:
            _beta_abs    = max(abs(float(betas.iloc[-1])), 0.01)
            _d_a         = STARTING_CAPITAL / (1.0 + _beta_abs)
            _d_b         = STARTING_CAPITAL - _d_a
            _entry_sa    = max(0.1, round(_d_a / float(open_entry_pa), 1))
            _entry_sb    = max(0.1, round(_d_b / float(open_entry_pb), 1))
        else:
            _entry_sa = _entry_sb = None

        open_trade = (
            {
                "direction":    open_direction,
                "entry_z":      open_entry_z,
                "entry_date":   open_entry_date,
                "curr_z":       curr_z,
                "pct_to_target": pct_to_target,
                "entry_price_a": open_entry_pa,
                "entry_price_b": open_entry_pb,
                "entry_shares_a": _entry_sa,   # LOCKED at entry
                "entry_shares_b": _entry_sb,   # LOCKED at entry
            }
            if in_open else None
        )

        direction_now = open_direction if open_trade else "NEUTRAL"

        info = {
            "pair":            str(ticker_a) + "/" + str(ticker_b),
            "a":               ticker_a,
            "b":               ticker_b,
            "price_a":         pair_df[ticker_a].iloc[-1],
            "price_b":         pair_df[ticker_b].iloc[-1],
            "curr_z":          curr_z,
            "beta":            betas.iloc[-1],
            "z_series":        z_series,
            "pair_df":         pair_df,       # needed for backtest
            "betas_series":    betas,         # needed for backtest
            "is_cointegrated": adf_pval < 0.05,
            "adf_pval":        adf_pval,
            "open_trade":      open_trade,
            "direction":       direction_now,
        }
        processed.append(info)
        if info["direction"] != "NEUTRAL":
            active.append(info)

    return processed, active


# =============================================================================
# 4. HTML HELPERS
# =============================================================================
def _row(label, value, val_color="#e8eaf0"):
    return (
        '<div style="display:flex;justify-content:space-between;align-items:center;'
        'padding:5px 0;border-bottom:1px solid rgba(255,255,255,0.04);">'
        '<span style="font-size:11px;color:#4a5568;font-family:monospace;">' + label + '</span>'
        '<span style="font-family:monospace;font-size:12px;font-weight:500;color:' + val_color + ';">' + value + '</span>'
        '</div>'
    )


# =============================================================================
# 5. ACTIVE SIGNAL CARD
# =============================================================================
def render_trade_card(sig):
    is_long     = sig["direction"] == "LONG"
    accent      = "#00ffcc" if is_long else "#ff4b4b"
    bg          = "rgba(0,255,204,0.04)"  if is_long else "rgba(255,75,75,0.04)"
    border      = "rgba(0,255,204,0.25)"  if is_long else "rgba(255,75,75,0.25)"
    leg1_action = "BUY"  if is_long else "SELL"
    leg2_action = "SELL" if is_long else "BUY"
    leg1_col    = "#00ffcc" if is_long else "#ff4b4b"
    leg2_col    = "#ff4b4b" if is_long else "#00ffcc"
    legs        = compute_legs(sig)
    ot          = sig.get("open_trade") or {}

    z_str       = str(round(sig["curr_z"], 2))
    beta_str    = str(round(sig["beta"], 2))
    entry_z_str = str(round(ot.get("entry_z", 0), 2)) if ot else "N/A"
    pct_raw     = ot.get("pct_to_target", 0) if ot else 0
    pct_clamped = max(0, min(float(pct_raw), 100))
    pct_str     = str(round(pct_clamped, 0)) + "%"
    pa_str      = "$" + str(round(sig["price_a"], 2))
    pb_str      = "$" + str(round(sig["price_b"], 2))
    sha_str     = str(legs["shares_a"])   # already float with 1dp
    shb_str     = str(legs["shares_b"])
    not_a_str   = "$" + "{:,.0f}".format(legs["notional_a"])
    not_b_str   = "$" + "{:,.0f}".format(legs["notional_b"])
    total_str   = "$" + "{:,.0f}".format(legs["total_cost"])
    hedge_str   = str(legs["shares_a"]) + ":" + str(legs["shares_b"])

    # Entry prices & P&L strings
    epa_str     = "$" + str(round(legs["entry_pa"], 2)) if legs["entry_pa"] else None
    epb_str     = "$" + str(round(legs["entry_pb"], 2)) if legs["entry_pb"] else None
    pnl_val     = legs["pnl"]
    pnl_a_val   = legs["pnl_a"]
    pnl_b_val   = legs["pnl_b"]
    has_pnl     = pnl_val is not None

    def fmt_pnl(v):
        if v is None: return ""
        sign = "+" if v >= 0 else ""
        col  = "#00d4a0" if v >= 0 else "#f56565"
        return '<span style="color:' + col + ';font-family:monospace;font-size:11px;font-weight:600;">' + sign + "${:,.0f}".format(v) + "</span>"

    pnl_badge = ""
    if has_pnl:
        pnl_sign  = "+" if pnl_val >= 0 else ""
        pnl_color = "#00d4a0" if pnl_val >= 0 else "#f56565"
        pnl_bg    = "rgba(0,212,160,0.10)" if pnl_val >= 0 else "rgba(245,101,101,0.10)"
        pnl_border= "rgba(0,212,160,0.30)" if pnl_val >= 0 else "rgba(245,101,101,0.30)"
        pnl_label = "UNREALISED P&L"
        pnl_pot     = legs.get("pnl_potential")
        pot_str     = ("  |  remaining: +" + "${:,.0f}".format(pnl_pot) if pnl_pot and pnl_pot > 0 else "")
        pnl_badge = (
            '<div style="background:' + pnl_bg + ';border:1px solid ' + pnl_border + ';'
            'border-radius:4px;padding:10px 14px;margin-bottom:14px;">'
            '<div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap;">'
            '<div>'
            '<p style="margin:0 0 2px;font-size:9px;color:#4a5568;font-family:monospace;'
            'text-transform:uppercase;letter-spacing:0.1em;">Spread P&L (Unrealised)</p>'
            '<span style="font-family:monospace;font-size:24px;font-weight:600;color:' + pnl_color + ';">'
            + pnl_sign + "${:,.0f}".format(pnl_val) + '</span>'
            + ('<span style="font-family:monospace;font-size:12px;color:#00d4a0;margin-left:8px;">'
               + pot_str + '</span>' if pot_str else "")
            + '</div>'
            '<div style="border-left:1px solid rgba(255,255,255,0.08);padding-left:12px;">'
            '<p style="margin:0 0 3px;font-size:9px;color:#4a5568;font-family:monospace;'
            'text-transform:uppercase;letter-spacing:0.1em;">Leg Breakdown</p>'
            '<p style="margin:0;font-size:11px;font-family:monospace;">'
            + sig["a"] + ': ' + fmt_pnl(pnl_a_val)
            + '  &nbsp; ' + sig["b"] + ': ' + fmt_pnl(pnl_b_val)
            + '</p>'
            '<p style="margin:4px 0 0;font-size:10px;color:#4a5568;font-family:monospace;">'
            'Measured as log-spread return × notional (beta-weighted)</p>'
            '</div>'
            '</div>'
            '</div>'
        )
    # ── Action label: context-aware based on Z progress ────────────
    abs_z = abs(float(sig["curr_z"]))
    if abs_z >= ENTRY_Z:
        action_label = ("BUY THE SPREAD — " + sig["a"] + " is Lagging") if is_long else ("SELL THE SPREAD — " + sig["a"] + " is Overextended")
    elif abs_z >= 1.0:
        action_label = "HOLD — Spread Narrowing, Approaching Target"
    else:
        action_label = "PREPARE TO EXIT — Z-Score Approaching 0.0"
    dir_label = action_label


    # ── Analyst note ─────────────────────────────────────────────────
    risk_ok    = abs(legs["risk_imbalance"]) < 20
    ratio_disp = legs["ratio"]
    ri_str     = ("+" if legs["risk_imbalance"] >= 0 else "") + "${:,.2f}".format(legs["risk_imbalance"])
    if is_long:
        spread_move = "narrowing ✓" if float(sig["curr_z"]) > float(ot.get("entry_z", sig["curr_z"])) else "still diverging"
        note_body = (
            "Spread is " + spread_move + ". Using ratio " + ratio_disp + " corrects the Price-Beta Trap — "
            + sig["b"] + " is no longer overpowering the trade. "
            + ("Risk imbalance " + ri_str + " is optimized. " if risk_ok else "Monitor risk imbalance " + ri_str + ". ")
            + "Exit when Z-Score hits 0.0."
        )
    else:
        spread_move = "narrowing ✓" if float(sig["curr_z"]) < float(ot.get("entry_z", sig["curr_z"])) else "still diverging"
        note_body = (
            "Spread is " + spread_move + ". Using ratio " + ratio_disp + " neutralizes the Zuckerberg Risk — "
            "selling fewer shares of the higher-priced " + sig["b"] + " keeps both legs dollar-balanced. "
            + ("Risk imbalance " + ri_str + " is optimized. " if risk_ok else "Monitor risk imbalance " + ri_str + ". ")
            + "Exit when Z-Score hits 0.0."
        )
    market_hedge = (
        "If the S&P 500 moves ±500pts, your " + sig["a"] + " move will be offset by " + sig["b"] + ". "
        "Your only alpha source is the Z-Score returning to 0."
    )
    analyst_note = (
        '<p style="margin:0 0 8px;font-size:11px;color:#e8eaf0;line-height:1.65;">' + note_body + '</p>'
        '<p style="margin:0 0 8px;font-size:11px;color:#8892a4;line-height:1.65;">' + market_hedge + '</p>'
        '<div style="display:flex;align-items:center;gap:8px;margin-top:10px;padding-top:8px;'
        'border-top:1px solid rgba(255,255,255,0.05);">'
        '<span style="font-size:9px;color:#f5a623;font-family:monospace;">⛔ STOP</span>'
        '<span style="font-size:10px;color:#4a5568;font-family:monospace;">Hard exit if |Z| hits ±' + str(STOP_Z) + '</span>'
        '<span style="font-size:9px;color:#00d4a0;font-family:monospace;margin-left:12px;">🎯 TARGET</span>'
        '<span style="font-size:10px;color:#4a5568;font-family:monospace;">Close both legs when Z = 0.0</span>'
        '</div>'
    )

    # ── Rebalance check ──────────────────────────────────────────────
    rb          = get_rebalance_instructions(sig)
    rb_status   = rb["status"]
    rb_color    = "#f5a623" if rb_status == "REBALANCE" else "#4a5568"
    rb_bg       = "rgba(245,166,23,0.08)" if rb_status == "REBALANCE" else "rgba(0,0,0,0)"
    rb_border   = "rgba(245,166,23,0.30)" if rb_status == "REBALANCE" else "rgba(255,255,255,0.04)"
    rb_icon     = "⚡" if rb_status == "REBALANCE" else "✓"

    if rb_status == "REBALANCE":
        rb_action_str = rb["action"] + " " + str(rb["qty"]) + " shs " + rb["ticker"]
        rb_notional   = "$" + "{:,.2f}".format(rb["notional"])
        rb_detail     = rb["action"] + " " + str(rb["qty"]) + " × " + pb_str + " = " + rb_notional
        rb_badge = (
            '<div style="background:' + rb_bg + ';border:1px solid ' + rb_border + ';'
            'border-radius:4px;padding:8px 12px;margin-bottom:10px;'
            'display:flex;justify-content:space-between;align-items:center;">'
            '<div>'
            '<p style="margin:0 0 2px;font-size:9px;color:#f5a623;font-family:monospace;'
            'text-transform:uppercase;letter-spacing:0.1em;">⚡ Rebalance Required · Day ' + str(rb["days_in"]) + '</p>'
            '<p style="margin:0;font-family:monospace;font-size:13px;font-weight:600;color:#f5a623;">'
            + rb_action_str + '</p>'
            '<p style="margin:2px 0 0;font-size:10px;color:#8892a4;font-family:monospace;">'
            + rb["reason"] + '</p>'
            '</div>'
            '<div style="text-align:right;">'
            '<p style="margin:0 0 2px;font-size:9px;color:#4a5568;font-family:monospace;">Cost</p>'
            '<p style="margin:0;font-family:monospace;font-size:13px;font-weight:600;color:#f5a623;">'
            + rb_notional + '</p>'
            '</div>'
            '</div>'
        )
    else:
        rb_badge = (
            '<div style="display:flex;align-items:center;gap:8px;'
            'padding:6px 10px;margin-bottom:10px;">'
            '<span style="font-size:10px;color:#00d4a0;font-family:monospace;">✓ Hedge Stable</span>'
            '<span style="font-size:10px;color:#4a5568;font-family:monospace;">' + rb["reason"] + '</span>'
            '</div>'
        )

    progress_bar = (
        '<div style="height:5px;background:rgba(255,255,255,0.07);border-radius:3px;margin-top:6px;">'
        '<div style="width:' + str(pct_clamped) + '%;height:100%;background:' + accent + ';border-radius:3px;'
        'transition:width 0.3s;"></div>'
        '</div>'
        '<div style="display:flex;justify-content:space-between;margin-top:4px;">'
        '<span style="font-size:9px;color:#4a5568;font-family:monospace;">Entry: ' + entry_z_str + '</span>'
        '<span style="font-size:9px;color:' + accent + ';font-family:monospace;font-weight:600;">'
        + pct_str + ' of the way to target</span>'
        '<span style="font-size:9px;color:#4a5568;font-family:monospace;">Target: 0.0</span>'
        '</div>'
    )

    html = (
        '<div style="background:' + bg + ';border:1px solid ' + border + ';'
        'border-top:3px solid ' + accent + ';padding:20px;border-radius:6px;margin-bottom:20px;">'

        '<div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:14px;">'
        '<div>'
        '<h2 style="margin:0 0 4px;color:' + accent + ';font-family:monospace;font-size:22px;">'
        + sig["a"] + ' <span style="color:#2d3748;font-weight:300;">/</span> ' + sig["b"] + '</h2>'
        '<span style="font-size:12px;color:#8892a4;font-family:monospace;">' + dir_label + '</span>'
        '</div>'
        '<div style="text-align:right;">'
        '<p style="margin:0 0 2px;font-size:10px;color:#4a5568;font-family:monospace;letter-spacing:0.1em;">Z-SCORE NOW</p>'
        '<p style="margin:0;font-family:monospace;font-size:30px;font-weight:600;color:' + accent + ';">' + z_str + '</p>'
        '<p style="margin:2px 0 0;font-size:10px;color:#4a5568;font-family:monospace;">hedge ratio β = ' + beta_str + '</p>'
        '</div>'
        '</div>'

        '<div style="background:rgba(0,0,0,0.2);border-radius:4px;padding:10px 14px;margin-bottom:14px;">'
        '<p style="margin:0 0 2px;font-size:10px;color:#4a5568;font-family:monospace;'
        'text-transform:uppercase;letter-spacing:0.1em;">Trade Progress — Distance to Profit Target (0.0)</p>'
        + progress_bar +
        '</div>'

        + '<div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;">'

        '<div style="background:rgba(0,0,0,0.25);border-radius:4px;padding:12px 14px;'
        'border:1px solid rgba(255,255,255,0.05);">'
        '<p style="margin:0 0 8px;font-size:10px;color:#4a9eff;font-family:monospace;'
        'text-transform:uppercase;letter-spacing:0.1em;">Execution — Fractional Shares (Medallion)</p>'
        + (pnl_badge if has_pnl else '')
        + _row(leg1_action + "  " + sig["a"],
               sha_str + " shs"
               + ("  entry " + epa_str + "  →  now " + pa_str if epa_str else "  @ " + pa_str)
               + "  =  " + not_a_str,
               leg1_col)
        + _row(leg2_action + "  " + sig["b"],
               shb_str + " shs"
               + ("  entry " + epb_str + "  →  now " + pb_str if epb_str else "  @ " + pb_str)
               + "  =  " + not_b_str,
               leg2_col)
        + _row("Total Deployed", total_str + "  of $" + str(STARTING_CAPITAL), "#e8eaf0")
        + _row("Share Ratio (β-neutral)",
               legs["ratio"] + "  (β=" + str(round(abs(sig["beta"]), 2)) + ")",
               "#e8c96d")
        + _row("Risk Imbalance (Δ-neutral check)",
               ("+" if legs["risk_imbalance"] >= 0 else "") + "${:,.2f}".format(legs["risk_imbalance"])
               + ("  ✓ neutral" if abs(legs["risk_imbalance"]) < 20 else "  ⚠ check sizing"),
               "#00d4a0" if abs(legs["risk_imbalance"]) < 20 else "#f5a623")
        + _row("Dollar Imbalance",
               ("+" if legs["imbalance"] >= 0 else "") + "${:,.2f}".format(legs["imbalance"])
               + ("  ✓ balanced" if legs["dollar_imbal"] < 20 else "  rebalance"),
               "#00d4a0" if legs["dollar_imbal"] < 20 else "#f5a623")
        + '<div style="margin-top:8px;border-top:1px solid rgba(255,255,255,0.05);padding-top:8px;">'
        + rb_badge
        + '</div>'
        + '</div>'

        '<div style="background:rgba(0,0,0,0.25);border-radius:4px;padding:12px 14px;'
        'border:1px solid rgba(255,255,255,0.05);">'
        '<p style="margin:0 0 10px;font-size:10px;color:#a78bfa;font-family:monospace;'
        'text-transform:uppercase;letter-spacing:0.1em;">Analyst Note</p>'
        + analyst_note
        + '</div>'

        '</div></div>'
    )

    st.markdown(html, unsafe_allow_html=True)


# =============================================================================
# 6. PAIR CHART
# =============================================================================
def render_pair_chart(p):
    z_data     = p["z_series"]
    open_trade = p.get("open_trade")
    is_long    = (open_trade or {}).get("direction") == "LONG"
    is_short   = (open_trade or {}).get("direction") == "SHORT"

    fig = go.Figure()

    # Active trade zone shading
    if open_trade:
        ez          = float(open_trade["entry_z"])
        y0          = min(ez, EXIT_Z)
        y1          = max(ez, EXIT_Z)
        shade_color = "rgba(0,255,204,0.10)" if is_long else "rgba(255,75,75,0.10)"
        border_col  = "rgba(0,255,204,0.35)" if is_long else "rgba(255,75,75,0.35)"
        fig.add_hrect(y0=y0, y1=y1, fillcolor=shade_color,
                      line=dict(color=border_col, width=1, dash="dot"), layer="below")

    # Z-score line
    fig.add_trace(go.Scatter(
        x=z_data.index, y=z_data,
        line=dict(color="#00d1ff", width=2.5),
        name="Z-Score", opacity=0.92,
        hovertemplate="%{x|%b %d %Y}  Z = %{y:.2f}<extra></extra>",
    ))

    # Glowing active entry marker
    if open_trade and open_trade.get("entry_date") in z_data.index:
        glow_col  = "#00ffcc" if is_long else "#ff4b4b"
        entry_dt  = open_trade["entry_date"]
        entry_y   = float(open_trade["entry_z"])
        glow_bg_o = "rgba(0,255,204,0.15)" if is_long else "rgba(255,75,75,0.15)"
        glow_bg_m = "rgba(0,255,204,0.30)" if is_long else "rgba(255,75,75,0.30)"
        glow_sym  = "triangle-up" if is_long else "triangle-down"

        fig.add_trace(go.Scatter(x=[entry_dt], y=[entry_y], mode="markers",
            marker=dict(color=glow_bg_o, size=32, symbol="circle"),
            showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=[entry_dt], y=[entry_y], mode="markers",
            marker=dict(color=glow_bg_m, size=21, symbol="circle"),
            showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=[entry_dt], y=[entry_y], mode="markers",
            marker=dict(color=glow_col, size=14, symbol=glow_sym,
                        line=dict(color="#0b0e14", width=2)),
            name="Active Entry",
            hovertemplate=(
                "<b>ACTIVE ENTRY (" + (open_trade["direction"] or "") + ")</b><br>"
                + entry_dt.strftime("%b %d, %Y")
                + "  Z at entry: " + str(round(entry_y, 2))
                + "<br>Z now: " + str(round(p["curr_z"], 2))
                + "<extra></extra>"
            ),
        ))
        fig.add_vline(x=entry_dt.isoformat(),
                      line=dict(color=glow_col, width=1.2, dash="dot"))

    # Threshold lines
    fig.add_hline(y=EXIT_Z, line=dict(color="#ffffff", width=1.2, dash="dot"),
                  annotation_text="TARGET / EXIT (0.0)",
                  annotation_font=dict(color="#8892a4", size=9),
                  annotation_position="bottom right")
    fig.add_hline(y=ENTRY_Z, line=dict(color="#ff4b4b", width=1.5),
                  annotation_text="SHORT ENTRY (+2.25)",
                  annotation_font=dict(color="#ff4b4b", size=9),
                  annotation_position="top left")
    fig.add_hline(y=-ENTRY_Z, line=dict(color="#00ffcc", width=1.5),
                  annotation_text="LONG ENTRY (-2.25)",
                  annotation_font=dict(color="#00ffcc", size=9),
                  annotation_position="bottom left")
    fig.add_hline(y=STOP_Z, line=dict(color="#f5a623", width=2.0, dash="dash"),
                  annotation_text="STOP LOSS (+3.5)",
                  annotation_font=dict(color="#f5a623", size=9),
                  annotation_position="top right")
    fig.add_hline(y=-STOP_Z, line=dict(color="#f5a623", width=2.0, dash="dash"),
                  annotation_text="STOP LOSS (-3.5)",
                  annotation_font=dict(color="#f5a623", size=9),
                  annotation_position="bottom right")

    # Axis ranges
    last_date    = z_data.index[-1]
    one_year_ago = last_date - pd.DateOffset(years=1)
    x_start      = max(one_year_ago, z_data.index[0])
    x_pad        = (last_date - x_start) * 0.12
    x_end        = last_date + x_pad
    visible_z    = z_data[z_data.index >= x_start]
    y_min        = float(visible_z.min())
    y_max        = float(visible_z.max())
    y_pad        = max((y_max - y_min) * 0.15, 0.5)
    y_lo         = min(y_min - y_pad, -STOP_Z - 0.4)
    y_hi         = max(y_max + y_pad,  STOP_Z + 0.4)

    # Status annotation
    annotations = []
    if open_trade:
        s_color  = "#00ffcc" if is_long else "#ff4b4b"
        pct_val  = float(open_trade.get("pct_to_target", 0))
        ann_txt  = (
            "<b>OPEN — " + str(open_trade["direction"]) + "</b><br>"
            "Z now: " + str(round(p["curr_z"], 2)) + "  Target: 0.0<br>"
            + str(round(pct_val, 0)) + "% of the way there"
        )
        annotations.append(dict(
            x=x_end, y=p["curr_z"], xref="x", yref="y",
            text=ann_txt, showarrow=True, arrowhead=2, arrowsize=0.9,
            arrowcolor=s_color, ax=-95, ay=0,
            font=dict(family="IBM Plex Mono", size=10, color=s_color),
            bgcolor="rgba(11,14,20,0.92)",
            bordercolor=s_color, borderwidth=1, borderpad=6, align="left",
        ))
    else:
        annotations.append(dict(
            x=x_end, y=p["curr_z"], xref="x", yref="y",
            text="NEUTRAL<br>Watching for ±" + str(ENTRY_Z),
            showarrow=False,
            font=dict(family="IBM Plex Mono", size=9, color="#4a5568"),
            bgcolor="rgba(11,14,20,0.85)",
            bordercolor="#1e2330", borderwidth=1, borderpad=5,
        ))

    fig.update_layout(
        template="plotly_dark", height=400,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
        showlegend=False, annotations=annotations,
        xaxis=dict(
            range=[x_start, x_end], showgrid=False,
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
        hovermode="x unified",
        font=dict(family="IBM Plex Mono"),
    )

    return fig


# =============================================================================
# 7. CHART FOOTER
# =============================================================================
def render_chart_footer(p):
    ot          = p.get("open_trade")
    coint_color = "#00ffcc" if p["is_cointegrated"] else "#f5a623"
    coint_label = "COINTEGRATED" if p["is_cointegrated"] else "DRIFTING"
    beta_str    = str(round(float(p["beta"]), 2))
    z_str       = str(round(float(p["curr_z"]), 2))
    adf_str     = "p=" + str(round(float(p["adf_pval"]), 3))

    if ot and ot.get("entry_date"):
        ot_color  = "#00ffcc" if ot["direction"] == "LONG" else "#ff4b4b"
        try:
            hold_days = (datetime.now().date() - ot["entry_date"].date()).days
        except Exception:
            hold_days = 0
        status_html = (
            '<span style="color:' + ot_color + ';font-family:monospace;font-size:11px;">'
            + ot["direction"] + ' OPEN — ' + str(hold_days) + 'd held'
            '</span>'
        )
    else:
        status_html = '<span style="color:#2d3748;font-family:monospace;font-size:11px;">Neutral</span>'

    # Rebalance chip in footer
    rb_footer = get_rebalance_instructions(p)
    if rb_footer["status"] == "REBALANCE":
        rb_chip = (
            '<span style="font-family:monospace;font-size:10px;padding:1px 7px;'
            'border-radius:3px;color:#f5a623;'
            'border:1px solid rgba(245,166,23,0.4);margin-left:8px;">'
            '⚡ Rebalance ' + rb_footer["action"] + " " + str(rb_footer["qty"]) + " " + rb_footer["ticker"]
            + '</span>'
        )
    else:
        rb_chip = ""

    st.markdown(
        '<div style="display:flex;justify-content:space-between;align-items:center;'
        'font-size:11px;margin-top:-12px;padding:0 6px 14px;">'
        '<span style="font-family:monospace;color:#4a5568;">β = <b style="color:#e8eaf0;">' + beta_str + '</b></span>'
        '<span style="font-family:monospace;color:#4a5568;">Z = <b style="color:#00d1ff;">' + z_str + '</b></span>'
        '<span style="font-family:monospace;color:' + coint_color + ';">' + coint_label + ' (' + adf_str + ')</span>'
        + status_html
        + rb_chip +
        '</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# 8. CHART LEGEND
# =============================================================================
def render_chart_legend():
    st.markdown(
        '<div style="display:flex;flex-wrap:wrap;gap:16px;padding:10px 4px 16px;'
        'font-family:monospace;font-size:11px;color:#4a5568;'
        'border-bottom:1px solid rgba(255,255,255,0.05);margin-bottom:16px;">'
        '<span><span style="color:#00ffcc;">──</span>  Long entry threshold</span>'
        '<span><span style="color:#ff4b4b;">──</span>  Short entry threshold</span>'
        '<span><span style="color:#ffffff;opacity:0.4;">┄┄</span>  Exit / profit target (0.0)</span>'
        '<span><span style="color:#f5a623;">─ ─</span>  Stop loss (±3.5)</span>'
        '<span><span style="color:#00ffcc;font-size:14px;">▲</span>/<span style="color:#ff4b4b;font-size:14px;">▼</span>  Active entry (glowing = position open)</span>'
        '<span><span style="background:rgba(0,255,204,0.15);padding:1px 6px;">■</span>  Teal zone = profit room left (Long)</span>'
        '<span><span style="background:rgba(255,75,75,0.15);padding:1px 6px;">■</span>  Red zone = profit room left (Short)</span>'
        '</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# 9. ENGINE EXPLAINER
# =============================================================================
def render_engine_explainer():
    st.markdown(
        '<div style="background:#111318;border-radius:6px;padding:22px 26px;'
        'border-left:3px solid #4a9eff;">'
        '<p style="font-family:monospace;font-size:12px;color:#4a9eff;margin:0 0 16px;'
        'text-transform:uppercase;letter-spacing:0.12em;">How the Engine Works — State Machine Hysteresis</p>'
        '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:22px;">'

        '<div>'
        '<p style="font-family:monospace;font-size:12px;color:#e8eaf0;margin:0 0 6px;">① Entry — Switch ON</p>'
        '<p style="font-size:12px;color:#8892a4;line-height:1.7;margin:0;">'
        'When Z crosses <b style="color:#e8eaf0;">±2.25</b> for the first time, '
        'a position opens. All subsequent re-crosses of the same threshold are '
        '<b style="color:#f5a623;">ignored</b>. '
        'No double-down. No averaging-in. One trade per pair at a time.'
        '</p></div>'

        '<div>'
        '<p style="font-family:monospace;font-size:12px;color:#e8eaf0;margin:0 0 6px;">② Noise Phase — Hold</p>'
        '<p style="font-size:12px;color:#8892a4;line-height:1.7;margin:0;">'
        'Z may bounce around the threshold — normal market noise. '
        'The shaded zone on each chart shows remaining profit room. '
        'The progress bar on the signal card shows how far the trade has moved '
        'toward the <b style="color:#ffffff;">0.0 target</b>.'
        '</p></div>'

        '<div>'
        '<p style="font-family:monospace;font-size:12px;color:#e8eaf0;margin:0 0 6px;">③ Exit — Switch OFF</p>'
        '<p style="font-size:12px;color:#8892a4;line-height:1.7;margin:0;">'
        'Position closes when Z crosses <b style="color:#ffffff;">0.0</b> '
        '(mean reversion complete — full profit) or hits '
        '<b style="color:#f5a623;">±3.5</b> (stop loss activated). '
        'Only after exit does the engine reset and scan for the next entry.'
        '</p></div>'

        '</div></div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# 9b. WATCHLIST PANEL  (neutral pairs)
# =============================================================================
def render_watchlist(neutral_pairs: list):
    if not neutral_pairs:
        return
    st.markdown(
        '<p style="font-family:monospace;font-size:10px;letter-spacing:0.12em;'
        'color:#4a5568;text-transform:uppercase;margin:0 0 10px;">Pair Watchlist — Neutral Zone</p>',
        unsafe_allow_html=True,
    )
    cols = st.columns(len(neutral_pairs))
    for i, p in enumerate(neutral_pairs):
        z    = float(p["curr_z"])
        dist = round(abs(abs(z) - ENTRY_Z), 2)
        z_color  = "#4a9eff" if z > 0 else "#00ffcc"
        coint_col= "#00d4a0" if p["is_cointegrated"] else "#f5a623"
        coint_lbl= "Cointegrated" if p["is_cointegrated"] else "Drifting"
        if z > 0:
            wait_msg = "Wait for +" + str(ENTRY_Z) + " to Short"
        elif abs(z) < 0.3:
            wait_msg = "Dormant — near zero"
        else:
            wait_msg = "Wait for -" + str(ENTRY_Z) + " to Buy"
        bar_pct = min(abs(z) / ENTRY_Z * 100, 100)
        bar_col = z_color
        cols[i].markdown(
            '<div style="background:#111318;border:1px solid rgba(255,255,255,0.07);'
            'border-radius:4px;padding:12px 14px;">'
            '<p style="margin:0 0 2px;font-family:monospace;font-size:13px;'
            'font-weight:500;color:#e8eaf0;">' + p["a"] + ' <span style="color:#2d3748;">/</span> ' + p["b"] + '</p>'
            '<p style="margin:0 0 8px;font-size:10px;color:#4a5568;font-family:monospace;">' + wait_msg + '</p>'
            '<div style="display:flex;align-items:baseline;gap:6px;margin-bottom:6px;">'
            '<span style="font-family:monospace;font-size:22px;font-weight:600;color:' + z_color + ';">' + str(round(z,2)) + '</span>'
            '<span style="font-size:10px;color:#4a5568;font-family:monospace;">/ ±' + str(ENTRY_Z) + '</span>'
            '</div>'
            '<div style="height:3px;background:rgba(255,255,255,0.07);border-radius:2px;margin-bottom:8px;">'
            '<div style="width:' + str(round(bar_pct,1)) + '%;height:100%;background:' + bar_col + ';'
            'border-radius:2px;"></div></div>'
            '<span style="font-size:9px;color:' + coint_col + ';'
            'font-family:monospace;">' + coint_lbl + '  ·  ' + str(dist) + 'σ from entry</span>'
            '</div>',
            unsafe_allow_html=True,
        )



# =============================================================================
# 10b. BACKTEST ENGINE  (2-year, state-machine, Medallion sizing)
# Rules:
#   Capital: $2,000 shared pool split equally per pair
#   Max simultaneous open trades: 3
#   Max hold: 21 trading days — hard close on day 21 regardless of Z
#   Entry: first Z cross of ±ENTRY_Z (toggle logic — no re-entries while open)
#   Exit:  Z returns to 0.0  |  stop ±STOP_Z  |  21-day timeout
#   P&L:   log-spread return × entry notional (Medallion beta-weighted)
# =============================================================================

import math as _math

def _bt_spread_pnl(ep_a, ep_b, cp_a, cp_b, sa, beta, direction):
    """Log-spread P&L — correct Medallion formula."""
    notional_a = sa * ep_a
    log_spread_entry = _math.log(ep_a) - beta * _math.log(ep_b)
    log_spread_now   = _math.log(cp_a) - beta * _math.log(cp_b)
    delta            = log_spread_now - log_spread_entry
    return round(delta * notional_a * (1 if direction == "LONG" else -1), 2)


def run_backtest(all_pairs_data: list) -> dict:
    """
    Runs a 2-year event-driven backtest across all pairs simultaneously.
    Returns portfolio equity curve, per-pair stats, and full trade log.
    """
    n       = len(all_pairs_data)
    alloc   = BT_CAPITAL / n

    # Build aligned 2-year price + z-score data per pair
    pair_data = {}
    for p in all_pairs_data:
        z   = p["z_series"].dropna()
        pdf = p["pair_df"] if "pair_df" in p else None
        bs  = p["betas_series"] if "betas_series" in p else None
        if pdf is None or bs is None:
            continue
        pdf = pdf.reindex(z.index).dropna()
        bs  = bs.reindex(z.index).ffill()
        common = z.index.intersection(pdf.index).intersection(bs.index)
        cutoff = common[-1] - pd.DateOffset(days=BT_LOOKBACK_DAYS)
        common = common[common >= cutoff]
        if len(common) < 60:
            continue
        pair_data[p["pair"]] = {
            "z": z.loc[common], "pdf": pdf.loc[common],
            "bs": bs.loc[common], "a": p["a"], "b": p["b"],
        }

    if not pair_data:
        return None

    # Union of all trading dates
    all_dates = sorted(set().union(*[d["z"].index for d in pair_data.values()]))

    # Per-pair state
    state = {
        pk: {
            "in_trade":   False, "direction": None,
            "entry_date": None,  "entry_z":   None,
            "entry_pa":   None,  "entry_pb":  None,
            "entry_sa":   None,  "entry_sb":  None,
            "trade_days": 0,     "equity":    alloc,
            "trades":     [],
        }
        for pk in pair_data
    }

    port_dates  = []
    port_equity = []

    for date in all_dates:
        # Count open positions across all pairs
        open_count = sum(1 for s in state.values() if s["in_trade"])

        for pk, pd_ in pair_data.items():
            z_s = pd_["z"]
            prices = pd_["pdf"]
            bs_s  = pd_["bs"]
            st    = state[pk]
            ta, tb = pd_["a"], pd_["b"]

            if date not in z_s.index:
                continue

            idx  = z_s.index.get_loc(date)
            curr = float(z_s.iloc[idx])
            prev = float(z_s.iloc[idx - 1]) if idx > 0 else curr
            pa   = float(prices[ta].loc[date])
            pb   = float(prices[tb].loc[date])
            beta = max(abs(float(bs_s.loc[date])), 0.01)

            if not st["in_trade"]:
                long_x  = prev > -ENTRY_Z and curr <= -ENTRY_Z
                short_x = prev < ENTRY_Z  and curr >= ENTRY_Z

                if (long_x or short_x) and open_count < BT_MAX_OPEN and st["equity"] > 10:
                    # Medallion entry-locked sizing
                    d_a  = st["equity"] / (1.0 + beta)
                    d_b  = st["equity"] - d_a
                    sa   = max(0.1, round(d_a / pa, 1))
                    sb   = max(0.1, round(d_b / pb, 1))
                    st.update({
                        "in_trade":   True,
                        "direction":  "LONG" if long_x else "SHORT",
                        "entry_date": date,
                        "entry_z":    curr,
                        "entry_pa":   pa,
                        "entry_pb":   pb,
                        "entry_sa":   sa,
                        "entry_sb":   sb,
                        "trade_days": 0,
                    })
                    open_count += 1

            else:
                st["trade_days"] += 1
                direction  = st["direction"]
                hit_exit   = (
                    (direction == "LONG"  and prev < 0 and curr >= 0) or
                    (direction == "SHORT" and prev > 0 and curr <= 0)
                )
                hit_stop   = abs(curr) >= STOP_Z
                hit_timeout= st["trade_days"] >= BT_MAX_HOLD_DAYS

                if hit_exit or hit_stop or hit_timeout:
                    pnl = _bt_spread_pnl(
                        st["entry_pa"], st["entry_pb"], pa, pb,
                        st["entry_sa"], beta, direction
                    )
                    st["equity"] += pnl

                    exit_reason = (
                        "TIMEOUT" if hit_timeout else
                        "STOP"    if hit_stop    else
                        "EXIT"
                    )
                    hold_days = (date - st["entry_date"]).days

                    st["trades"].append({
                        "pair":        pk,
                        "entry_date":  st["entry_date"],
                        "exit_date":   date,
                        "direction":   direction,
                        "pnl":         pnl,
                        "hold_days":   hold_days,
                        "trade_days":  st["trade_days"],
                        "exit_reason": exit_reason,
                        "entry_z":     st["entry_z"],
                        "exit_z":      curr,
                    })

                    st.update({
                        "in_trade":   False, "direction": None,
                        "entry_date": None,  "entry_z":   None,
                        "entry_pa":   None,  "entry_pb":  None,
                        "entry_sa":   None,  "entry_sb":  None,
                        "trade_days": 0,
                    })
                    open_count = max(0, open_count - 1)

        port_val = sum(s["equity"] for s in state.values())
        port_dates.append(date)
        port_equity.append(port_val)

    port_series = pd.Series(port_equity, index=port_dates).groupby(level=0).last()

    # Aggregate stats
    all_trades = []
    per_pair   = {}
    for pk, st in state.items():
        tds = st["trades"]
        all_trades.extend(tds)
        eq  = alloc  # fallback if no trades
        if tds:
            # Reconstruct equity curve from trades
            eq_dates  = [all_dates[0]]
            eq_vals   = [alloc]
            running   = alloc
            for t in tds:
                running += t["pnl"]
                eq_dates.append(t["exit_date"])
                eq_vals.append(running)
            eq = running
        tdf = pd.DataFrame(tds) if tds else pd.DataFrame()
        per_pair[pk] = {
            "num_trades":   len(tds),
            "num_wins":     int((tdf["pnl"] > 0).sum()) if len(tdf) else 0,
            "num_losses":   int((tdf["pnl"] <= 0).sum()) if len(tdf) else 0,
            "total_pnl":    round(tdf["pnl"].sum(), 2) if len(tdf) else 0,
            "win_rate":     round((tdf["pnl"] > 0).mean() * 100, 1) if len(tdf) else 0,
            "avg_hold":     round(tdf["hold_days"].mean(), 1) if len(tdf) else 0,
            "avg_trade_days":round(tdf["trade_days"].mean(), 1) if len(tdf) else 0,
            "best":         round(tdf["pnl"].max(), 2) if len(tdf) else 0,
            "worst":        round(tdf["pnl"].min(), 2) if len(tdf) else 0,
            "timeouts":     int((tdf["exit_reason"] == "TIMEOUT").sum()) if len(tdf) else 0,
            "stops":        int((tdf["exit_reason"] == "STOP").sum()) if len(tdf) else 0,
            "final_equity": round(eq, 2),
        }

    all_tdf = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()
    days_span = max((port_series.index[-1] - port_series.index[0]).days, 1)
    final_eq  = port_series.iloc[-1]
    port_apy  = round(((final_eq / BT_CAPITAL) ** (365 / days_span) - 1) * 100, 1)
    roll_max  = port_series.cummax()
    port_dd   = round(((port_series - roll_max) / roll_max * 100).min(), 1)
    total_pnl = round(all_tdf["pnl"].sum(), 2) if len(all_tdf) else 0
    total_wins  = int((all_tdf["pnl"] > 0).sum()) if len(all_tdf) else 0
    total_losses= int((all_tdf["pnl"] <= 0).sum()) if len(all_tdf) else 0
    port_wr   = round(total_wins / len(all_tdf) * 100, 1) if len(all_tdf) else 0
    avg_hold  = round(all_tdf["hold_days"].mean(), 1) if len(all_tdf) else 0
    timeouts  = int((all_tdf["exit_reason"] == "TIMEOUT").sum()) if len(all_tdf) else 0
    stops     = int((all_tdf["exit_reason"] == "STOP").sum()) if len(all_tdf) else 0

    return {
        "portfolio_equity": port_series,
        "per_pair":         per_pair,
        "all_trades":       all_trades,
        "all_tdf":          all_tdf,
        "total_trades":     len(all_trades),
        "total_wins":       total_wins,
        "total_losses":     total_losses,
        "total_pnl":        total_pnl,
        "port_wr":          port_wr,
        "port_apy":         port_apy,
        "port_dd":          port_dd,
        "avg_hold":         avg_hold,
        "timeouts":         timeouts,
        "stops":            stops,
        "final_equity":     round(final_eq, 2),
    }


def render_backtest(all_pairs_data: list):
    """Full backtest dashboard section."""
    st.divider()
    st.markdown(
        '<div style="display:flex;align-items:center;gap:14px;margin-bottom:4px;">'
        '<h2 style="margin:0;font-family:monospace;">Backtest</h2>'
        '<span style="font-family:monospace;font-size:10px;padding:3px 10px;border-radius:3px;'
        'color:#a78bfa;border:1px solid rgba(167,139,250,0.4);">'
        '$2,000 · 2yr · max 3 open · 21-day timeout</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.spinner("Running 2-year state-machine backtest..."):
        bt = run_backtest(all_pairs_data)

    if bt is None:
        st.warning("Insufficient data for backtest (need pair_df + betas_series in process_pairs).")
        return

    port_eq = bt["portfolio_equity"]

    # ── KPI strip ──────────────────────────────────────────────────────────
    def kpi(col, label, val, color="#e8eaf0"):
        col.markdown(
            '<div style="background:#111318;padding:14px 16px;border-radius:4px;'
            'border:1px solid rgba(255,255,255,0.07);text-align:center;">'
            '<p style="margin:0 0 4px;font-size:9px;color:#4a5568;font-family:monospace;'
            'text-transform:uppercase;letter-spacing:0.1em;">' + label + '</p>'
            '<p style="margin:0;font-family:monospace;font-size:20px;font-weight:600;color:'
            + color + ';">' + val + '</p></div>',
            unsafe_allow_html=True,
        )

    pnl_c = "#00d4a0" if bt["total_pnl"] >= 0 else "#f56565"
    apy_c = "#00d4a0" if bt["port_apy"]  >= 0 else "#f56565"
    dd_c  = "#f5a623" if bt["port_dd"]   > -15 else "#f56565"

    k = st.columns(8)
    kpi(k[0], "Total P&L",    f"${bt['total_pnl']:+,.0f}",          pnl_c)
    kpi(k[1], "APY",          f"{bt['port_apy']:+.1f}%",             apy_c)
    kpi(k[2], "Win Rate",     f"{bt['port_wr']:.0f}%",              "#e8c96d")
    kpi(k[3], "Total Trades", str(bt["total_trades"]),               "#e8eaf0")
    kpi(k[4], "W / L",        f"{bt['total_wins']} / {bt['total_losses']}", "#e8eaf0")
    kpi(k[5], "Max Drawdown", f"{bt['port_dd']:.1f}%",              dd_c)
    kpi(k[6], "Avg Hold",     f"{bt['avg_hold']:.0f}d",             "#a78bfa")
    kpi(k[7], "Timeouts",     str(bt["timeouts"]),                   "#f5a623" if bt["timeouts"] > 0 else "#4a5568")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Portfolio equity chart ─────────────────────────────────────────────
    pair_colors = ["#4a9eff", "#a78bfa", "#f5a623", "#e8c96d", "#f687b3"]
    fig = go.Figure()

    # Individual pair equity curves (reconstructed)
    for i, (pk, ps) in enumerate(bt["per_pair"].items()):
        tds = [t for t in bt["all_trades"] if t["pair"] == pk]
        if not tds:
            continue
        eq_d = [port_eq.index[0]]
        eq_v = [BT_ALLOC_PER]
        running = BT_ALLOC_PER
        for t in tds:
            running += t["pnl"]
            eq_d.append(t["exit_date"])
            eq_v.append(running)
        fig.add_trace(go.Scatter(
            x=eq_d, y=eq_v, mode="lines",
            line=dict(color=pair_colors[i % len(pair_colors)], width=1.2),
            opacity=0.4, name=pk,
            hovertemplate="<b>" + pk + "</b><br>%{x|%b %d %Y}<br>$%{y:,.0f}<extra></extra>",
        ))

    # Portfolio total
    fig.add_trace(go.Scatter(
        x=port_eq.index, y=port_eq,
        fill="tozeroy", fillcolor="rgba(0,212,160,0.06)",
        line=dict(color="#00d4a0", width=3), name="Portfolio Total",
        hovertemplate="<b>Portfolio</b><br>%{x|%b %d %Y}<br>$%{y:,.0f}<extra></extra>",
    ))

    # Starting capital line
    fig.add_hline(
        y=BT_CAPITAL,
        line=dict(color="rgba(255,255,255,0.18)", width=1, dash="dot"),
        annotation_text="Starting Capital $" + f"{BT_CAPITAL:,}",
        annotation_font_color="#4a5568", annotation_font_size=10,
        annotation_position="top left",
    )

    # Trade markers on portfolio curve
    if bt["all_trades"]:
        atdf = bt["all_tdf"]
        e_dates = atdf["entry_date"].tolist()
        e_vals  = [port_eq.asof(d) if d >= port_eq.index[0] else None for d in e_dates]
        e_cols  = ["#00d4a0" if d == "LONG" else "#f56565" for d in atdf["direction"].tolist()]
        fig.add_trace(go.Scatter(
            x=e_dates, y=e_vals, mode="markers",
            marker=dict(color=e_cols, size=8, symbol="circle",
                        line=dict(color="#0b0e14", width=1.5)),
            name="Entry",
            hovertemplate="<b>ENTRY</b> %{x|%b %d %Y}<extra></extra>",
        ))

        x_dates = atdf["exit_date"].tolist()
        x_vals  = [port_eq.asof(d) if d >= port_eq.index[0] else None for d in x_dates]
        x_cols  = [
            "#f5a623" if r == "STOP" else "#f56565" if r == "TIMEOUT" else "#ffffff"
            for r in atdf["exit_reason"].tolist()
        ]
        x_syms  = [
            "x" if r == "STOP" else "diamond-open" if r == "TIMEOUT" else "diamond"
            for r in atdf["exit_reason"].tolist()
        ]
        fig.add_trace(go.Scatter(
            x=x_dates, y=x_vals, mode="markers",
            marker=dict(color=x_cols, size=8, symbol=x_syms,
                        line=dict(color="#0b0e14", width=1.5)),
            name="Exit",
            hovertemplate="<b>EXIT</b> %{x|%b %d %Y}<extra></extra>",
        ))

    y_lo = min(port_eq.min() * 0.97, BT_CAPITAL * 0.92) if len(port_eq) else 0
    y_hi = max(port_eq.max() * 1.03, BT_CAPITAL * 1.08) if len(port_eq) else BT_CAPITAL * 1.1

    fig.update_layout(
        template="plotly_dark", height=420,
        paper_bgcolor="#0b0e14", plot_bgcolor="#0b0e14",
        margin=dict(l=12, r=12, t=12, b=12),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
                    font=dict(family="IBM Plex Mono", size=10, color="#8892a4"),
                    bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(showgrid=False,
                   rangeselector=dict(
                       bgcolor="#111318", activecolor="#00d4a0",
                       bordercolor="rgba(255,255,255,0.1)",
                       font=dict(family="IBM Plex Mono", size=10, color="#8892a4"),
                       buttons=[
                           dict(count=6, label="6M", step="month", stepmode="backward"),
                           dict(count=1, label="1Y", step="year",  stepmode="backward"),
                           dict(step="all", label="All"),
                       ],
                   )),
        yaxis=dict(showgrid=False, zeroline=False, tickprefix="$", range=[y_lo, y_hi]),
        hovermode="x unified", font=dict(family="IBM Plex Mono"),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Legend
    st.markdown(
        '<div style="display:flex;gap:20px;padding:0 4px 16px;font-family:monospace;'
        'font-size:11px;color:#4a5568;">'
        '<span><span style="color:#00d4a0;">●</span> Long entry</span>'
        '<span><span style="color:#f56565;">●</span> Short entry</span>'
        '<span><span style="color:#ffffff;">◆</span> Exit (target)</span>'
        '<span><span style="color:#f5a623;">✕</span> Stop loss</span>'
        '<span><span style="color:#f56565;">◇</span> Timeout (21d)</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Per-pair cards ─────────────────────────────────────────────────────
    st.markdown(
        '<p style="font-family:monospace;font-size:10px;letter-spacing:0.12em;'
        'color:#4a5568;text-transform:uppercase;margin:4px 0 14px;">Per-Pair Performance</p>',
        unsafe_allow_html=True,
    )

    def row(label, val, col="#e8eaf0"):
        return (
            '<div style="display:flex;justify-content:space-between;margin-bottom:4px;">'
            '<span style="font-size:10px;color:#4a5568;font-family:monospace;">' + label + '</span>'
            '<span style="font-family:monospace;font-size:11px;color:' + col + ';">' + val + '</span>'
            '</div>'
        )

    pcols = st.columns(len(bt["per_pair"]))
    for i, (pk, ps) in enumerate(bt["per_pair"].items()):
        pc = "#00d4a0" if ps["total_pnl"] >= 0 else "#f56565"
        ac = "#00d4a0" if ps["total_pnl"] >= 0 else "#f56565"
        pcols[i].markdown(
            '<div style="background:#111318;padding:12px 14px;border-radius:4px;'
            'border:1px solid rgba(255,255,255,0.07);">'
            '<p style="margin:0 0 8px;font-family:monospace;font-size:12px;'
            'font-weight:600;color:#e8eaf0;">' + pk + '</p>'
            + row("P&L",       f"${ps['total_pnl']:+,.0f}", pc)
            + row("Alloc",     f"${BT_ALLOC_PER:,.0f}")
            + row("Final",     f"${ps['final_equity']:,.0f}", pc)
            + row("Trades",    str(ps["num_trades"]))
            + row("Win Rate",  f"{ps['win_rate']:.0f}%", "#e8c96d")
            + row("Avg Hold",  f"{ps['avg_hold']:.0f}d", "#a78bfa")
            + row("Avg Days",  f"{ps['avg_trade_days']:.0f}d")
            + row("Timeouts",  str(ps["timeouts"]), "#f5a623" if ps["timeouts"] else "#4a5568")
            + row("Stops",     str(ps["stops"]),    "#f56565" if ps["stops"] else "#4a5568")
            + row("Best",      f"${ps['best']:+,.0f}", "#00d4a0")
            + row("Worst",     f"${ps['worst']:+,.0f}", "#f56565")
            + '</div>',
            unsafe_allow_html=True,
        )

    # ── Rule explainer ──────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        '<div style="background:#111318;border-radius:6px;padding:18px 22px;'
        'border-left:3px solid #a78bfa;">'
        '<p style="font-family:monospace;font-size:10px;color:#a78bfa;margin:0 0 12px;'
        'text-transform:uppercase;letter-spacing:0.1em;">Backtest Rules</p>'
        '<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;">'
        '<div><p style="font-family:monospace;font-size:11px;color:#e8eaf0;margin:0 0 4px;">Capital</p>'
        '<p style="font-size:11px;color:#8892a4;margin:0;line-height:1.6;">'
        f'${BT_CAPITAL:,} shared pool split equally across {len(PAIRS)} pairs (${BT_ALLOC_PER:,.0f}/pair). '
        'If all slots filled, new entries wait.</p></div>'
        '<div><p style="font-family:monospace;font-size:11px;color:#e8eaf0;margin:0 0 4px;">Max Open</p>'
        '<p style="font-size:11px;color:#8892a4;margin:0;line-height:1.6;">'
        f'Max {BT_MAX_OPEN} trades open simultaneously. Prevents over-concentration '
        'and models realistic capital constraints.</p></div>'
        '<div><p style="font-family:monospace;font-size:11px;color:#e8eaf0;margin:0 0 4px;">21-Day Timeout</p>'
        '<p style="font-size:11px;color:#8892a4;margin:0;line-height:1.6;">'
        'If Z has not returned to 0.0 within 21 trading days, both legs close at market. '
        'Frees capital for better opportunities.</p></div>'
        '<div><p style="font-family:monospace;font-size:11px;color:#e8eaf0;margin:0 0 4px;">P&L Method</p>'
        '<p style="font-size:11px;color:#8892a4;margin:0;line-height:1.6;">'
        'Log-spread return × entry notional (Medallion beta-weighted). '
        'Measures spread reversion correctly regardless of market direction.</p></div>'
        '</div></div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# 10. MAIN TERMINAL
# =============================================================================
def main():
    col_title, col_meta = st.columns([3, 1])
    with col_title:
        st.title("Omni-Arb Terminal v6.0")
        st.caption(
            "Universe: S&P 500 Sector Pairs  |  Engine: State-Machine Cointegration  |  "
            "Capital: $" + str(STARTING_CAPITAL) + " per signal  |  "
            "Refreshed: " + datetime.now().strftime("%b %d %Y %H:%M ET")
        )
    with col_meta:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            '<div style="text-align:right;font-family:monospace;font-size:10px;color:#4a5568;line-height:1.8;">'
            'Entry threshold  ±' + str(ENTRY_Z) + '<br>'
            'Stop loss  ±' + str(STOP_Z) + '<br>'
            'Profit target  0.0<br>'
            'Rolling window  ' + str(ROLLING_WINDOW) + 'd'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    with st.spinner("Syncing market data..."):
        data_raw = get_market_data()

    all_pairs, active_pairs = process_pairs(data_raw)

    n_active  = len(active_pairs)
    n_neutral = len(all_pairs) - n_active
    s_color   = "#00ffcc" if n_active > 0 else "#4a5568"

    st.markdown(
        '<div style="display:flex;align-items:center;gap:16px;margin-bottom:18px;">'
        '<h2 style="margin:0;font-family:monospace;">Active Signals</h2>'
        '<span style="font-family:monospace;font-size:12px;padding:4px 12px;border-radius:3px;'
        'color:' + s_color + ';border:1px solid ' + s_color + ';">'
        + str(n_active) + ' open  /  ' + str(n_neutral) + ' neutral'
        '</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    if active_pairs:
        for sig in active_pairs:
            render_trade_card(sig)
    else:
        st.markdown(
            '<div style="background:#111318;border:1px dashed rgba(255,255,255,0.08);'
            'border-radius:6px;padding:32px;text-align:center;">'
            '<p style="font-family:monospace;font-size:13px;color:#4a5568;margin:0 0 8px;">'
            'No active deviations — all pairs within ±' + str(ENTRY_Z) + 'σ'
            '</p>'
            '<p style="font-family:monospace;font-size:11px;color:#2d3748;margin:0;">'
            'Engine is live. Signals appear the moment Z crosses the entry threshold.'
            '</p>'
            '</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Watchlist: neutral pairs
    neutral_pairs = [p for p in all_pairs if p["direction"] == "NEUTRAL"]
    if neutral_pairs:
        render_watchlist(neutral_pairs)
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div style="display:flex;align-items:baseline;gap:12px;margin-bottom:8px;">'
        '<h2 style="margin:0;font-family:monospace;">Pair Analysis</h2>'
        '<span style="font-family:monospace;font-size:11px;color:#4a5568;">'
        '1-year default view — use range buttons to zoom in/out'
        '</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    render_chart_legend()

    chart_cols = st.columns(2)
    for i, p in enumerate(all_pairs):
        with chart_cols[i % 2]:
            ot     = p.get("open_trade")
            accent = (
                "#00ffcc" if (ot and ot["direction"] == "LONG") else
                "#ff4b4b" if (ot and ot["direction"] == "SHORT") else
                "#4a5568"
            )
            badge = (
                '<span style="font-family:monospace;font-size:9px;font-weight:600;'
                'letter-spacing:0.12em;padding:2px 8px;border-radius:2px;'
                'color:' + accent + ';border:1px solid ' + accent + ';opacity:0.85;">'
                + ot["direction"] + ' OPEN</span>'
                if ot else
                '<span style="font-family:monospace;font-size:9px;color:#2d3748;">NEUTRAL</span>'
            )
            st.markdown(
                '<div style="display:flex;justify-content:space-between;'
                'align-items:center;margin-bottom:6px;">'
                '<p style="margin:0;font-family:monospace;font-size:15px;font-weight:500;'
                'color:' + accent + ';">'
                + p["a"] + ' <span style="color:#2d3748;font-weight:300;">/</span> ' + p["b"]
                + '</p>' + badge + '</div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(render_pair_chart(p), use_container_width=True)
            render_chart_footer(p)

    st.divider()
    render_engine_explainer()

    # Backtest section
    render_backtest(all_pairs)


if __name__ == "__main__":
    main()

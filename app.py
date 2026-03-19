# ==========================================
# ENHANCED: ACTIVE TRADE SUMMARY — INSTITUTIONAL GRADE
# Renaissance / Medallion-inspired stat arb dashboard
# ==========================================

import streamlit as st
import numpy as np

# ── CONFIG ──────────────────────────────────────────────────
ENTRY_Z   = 2.0
EXIT_Z    = 0.5
MAX_Z_VIZ = 3.5   # normalization ceiling for z-bar

# ── SIGNAL QUALITY SCORER ───────────────────────────────────
def score_signal(sig: dict) -> int:
    """
    Composite 0–5 signal confidence score.
    Mirrors Medallion-style multi-factor confirmation:
      cointegration pass, ADF p-val, z-magnitude, R², half-life speed
    """
    score = 0
    if sig["is_cointegrated"]:           score += 2   # primary gating criterion
    if sig["adf_pval"] < 0.05:           score += 1   # statistical rigor
    if abs(sig["curr_z"]) > 2.5:        score += 1   # extreme deviation = higher edge
    if sig.get("r_squared", 0) > 0.80:  score += 1   # tight hedge proxy
    return min(score, 5)

# ── EXECUTION SIZING (dollar-neutral) ───────────────────────
def compute_legs(sig: dict) -> dict:
    shares_a = sig["shares"]
    shares_b = int(round(shares_a * sig["beta"]))
    notional_a = shares_a * sig["price_a"]
    notional_b = shares_b * sig["price_b"]
    dollar_imbalance = notional_a - notional_b   # target: ~0
    return {
        "shares_a":  shares_a,
        "shares_b":  shares_b,
        "notional_a": notional_a,
        "notional_b": notional_b,
        "imbalance":  dollar_imbalance,
    }

# ── CARD RENDERER ────────────────────────────────────────────
def render_trade_card(sig: dict) -> str:
    is_long       = sig["direction"] == "LONG"
    accent        = "#00d4a0" if is_long else "#f56565"
    bg_accent     = "rgba(0,212,160,0.06)" if is_long else "rgba(245,101,101,0.05)"
    dir_label     = sig["direction"]
    z_meaning     = "Spread undervalued — buy A / sell B" if is_long else "Spread overvalued — sell A / buy B"

    # Z-score bar (bidirectional)
    z_pct = min(abs(sig["curr_z"]) / MAX_Z_VIZ, 1.0) * 50  # half of 100%
    bar_left  = f"{50 - z_pct:.1f}%" if is_long else "50%"
    bar_width = f"{z_pct:.1f}%"

    # Execution legs
    legs = compute_legs(sig)
    leg1_verb, leg2_verb = ("BUY", "SELL") if is_long else ("SELL", "BUY")
    leg1_col  = "#00d4a0" if is_long else "#f56565"
    leg2_col  = "#f56565" if is_long else "#00d4a0"

    exec_html = f"""
      <div style="background:#181c24;border-radius:3px;padding:10px 12px;margin-bottom:12px;">
        <p style="margin:0 0 8px;font-size:9px;color:#4a5568;font-family:monospace;
                  text-transform:uppercase;letter-spacing:0.1em;">Execution (Dollar-Neutral)</p>
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">
          <span style="font-family:monospace;font-size:11px;font-weight:600;color:{leg1_col};
                background:{"rgba(0,212,160,0.12)" if is_long else "rgba(245,101,101,0.1)"};
                padding:2px 7px;border-radius:2px;">{leg1_verb}</span>
          <span style="font-family:monospace;font-size:12px;color:#e8eaf0;padding:0 8px;">{sig['a']}</span>
          <span style="font-family:monospace;font-size:11px;color:#8892a4;">{legs['shares_a']} shs @ ${sig['price_a']:.2f}</span>
          <span style="font-family:monospace;font-size:11px;color:#4a5568;">${legs['notional_a']:,.0f}</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;">
          <span style="font-family:monospace;font-size:11px;font-weight:600;color:{leg2_col};
                background:{"rgba(245,101,101,0.1)" if is_long else "rgba(0,212,160,0.12)"};
                padding:2px 7px;border-radius:2px;">{leg2_verb}</span>
          <span style="font-family:monospace;font-size:12px;color:#e8eaf0;padding:0 8px;">{sig['b']}</span>
          <span style="font-family:monospace;font-size:11px;color:#8892a4;">{legs['shares_b']} shs @ ${sig['price_b']:.2f}</span>
          <span style="font-family:monospace;font-size:11px;color:#4a5568;">${legs['notional_b']:,.0f}</span>
        </div>
        <p style="margin:8px 0 0;font-size:9px;color:#4a5568;font-family:monospace;text-align:right;">
          Net imbalance: <span style="color:{"#00d4a0" if abs(legs['imbalance'])<500 else "#f5a623"}">
            ${abs(legs['imbalance']):,.0f}</span>
        </p>
      </div>"""

    # Signal quality pips
    score  = score_signal(sig)
    pip_on = "#00d4a0" if (is_long and sig["is_cointegrated"]) else ("#f56565" if not is_long else "#f5a623")
    pips   = "".join([
        f'<div style="width:14px;height:4px;border-radius:1px;background:{"" + pip_on if i < score else "#1e2330"};"></div>'
        for i in range(5)
    ])

    coint_color = "#00d4a0" if sig["is_cointegrated"] else "#f5a623"
    coint_text  = f"Cointegrated (p={sig['adf_pval']:.3f})" if sig["is_cointegrated"] else f"NOT cointegrated (p={sig['adf_pval']:.2f})"

    warn_banner = "" if sig["is_cointegrated"] else f"""
      <div style="background:rgba(245,166,23,0.07);border-top:1px solid rgba(245,166,23,0.2);
                  padding:7px 16px;display:flex;align-items:center;gap:8px;">
        <span style="font-family:monospace;font-size:10px;font-weight:600;color:#f5a623;">!</span>
        <span style="font-family:monospace;font-size:10px;color:rgba(245,166,23,0.8);">
          Pair not cointegrated — reduce sizing, widen stop by 0.5σ
        </span>
      </div>"""

    hl    = sig.get("half_life",    "–")
    r2    = sig.get("r_squared",    None)
    edge  = sig.get("edge_bps",     "–")
    sharpe= sig.get("sharpe_est",   None)
    lkbk  = sig.get("lookback",     63)
    sharpe_color = "#00d4a0" if (sharpe and sharpe > 1.5) else "#f5a623"

    metrics_html = f"""
      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:14px;">
        {"".join([
          f'<div style="background:#181c24;padding:8px 10px;border-radius:3px;">'
          f'<p style="margin:0 0 3px;font-size:9px;color:#4a5568;font-family:monospace;text-transform:uppercase;letter-spacing:0.08em;">{lab}</p>'
          f'<p style="margin:0;font-family:monospace;font-size:13px;font-weight:500;color:{col};">{val}</p></div>'
          for lab, val, col in [
            ("Half-Life",  f"{hl:.1f}d" if isinstance(hl, float) else hl, "#e8eaf0"),
            ("R²",         f"{r2*100:.0f}%" if r2 else "–", "#e8eaf0"),
            ("Edge",       f"{edge}bps",                                   "#e8c96d"),
            ("Sharpe Est", f"{sharpe:.1f}" if sharpe else "–",             sharpe_color),
            ("ADF p-val",  f"{sig['adf_pval']:.3f}",                       "#00d4a0" if sig['adf_pval'] < 0.05 else "#f5a623"),
            ("Lookback",   f"{lkbk}d",                                     "#e8eaf0"),
          ]
        ])}
      </div>"""

    return f"""
    <div style="background:#111318;border-radius:4px;border:1px solid {"rgba(255,255,255,0.07)"};
                overflow:hidden;margin-bottom:12px;position:relative;
                border-top:2px solid {accent};background:{bg_accent};">
      <div style="padding:14px 16px 12px;border-bottom:1px solid rgba(255,255,255,0.07);
                  display:flex;justify-content:space-between;align-items:flex-start;">
        <div>
          <p style="margin:0;font-family:monospace;font-size:15px;font-weight:600;
                    letter-spacing:0.04em;color:{accent};">
            {sig['a']} <span style="color:#4a5568;font-weight:300;">/</span> {sig['b']}
          </p>
          <p style="margin:3px 0 0;font-size:10px;color:#4a5568;font-family:monospace;">
            β={sig['beta']:.2f} · {lkbk}d lookback
          </p>
        </div>
        <span style="font-family:monospace;font-size:9px;font-weight:600;letter-spacing:0.14em;
              padding:3px 8px;border-radius:2px;
              background:{"rgba(0,212,160,0.15)" if is_long else "rgba(245,101,101,0.15)"};
              color:{accent};border:1px solid {"rgba(0,212,160,0.3)" if is_long else "rgba(245,101,101,0.3)"};">
          {dir_label}
        </span>
      </div>

      <div style="padding:14px 16px;">
        <div style="display:flex;align-

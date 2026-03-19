# =============================================================================
# OMNI-ARB  |  Statistical Arbitrage Active Trade Dashboard
# Institutional-grade pair trading signal monitor
# Compatible: Streamlit >= 1.28  |  Python >= 3.9
# =============================================================================

import streamlit as st
from datetime import datetime

# =============================================================================
# CONFIG
# =============================================================================
ENTRY_Z   = 2.0
EXIT_Z    = 0.5
MAX_Z_VIZ = 3.5


# =============================================================================
# MOCK SIGNALS  —  replace get_active_signals() with your live feed
# =============================================================================
MOCK_SIGNALS = [
    {
        "a": "JPM",  "b": "BAC",   "direction": "LONG",
        "curr_z": -2.41, "price_a": 198.32, "price_b": 44.17,  "beta": 4.31,
        "shares": 100,   "is_cointegrated": True,  "adf_pval": 0.021,
        "half_life": 8.3,  "r_squared": 0.87, "lookback": 63,
        "edge_bps": 34,  "sharpe_est": 1.8,
    },
    {
        "a": "XOM",  "b": "CVX",   "direction": "SHORT",
        "curr_z": 2.78,  "price_a": 112.50, "price_b": 158.90, "beta": 0.68,
        "shares": 100,   "is_cointegrated": True,  "adf_pval": 0.008,
        "half_life": 12.1, "r_squared": 0.91, "lookback": 126,
        "edge_bps": 28,  "sharpe_est": 2.1,
    },
    {
        "a": "MSFT", "b": "GOOGL", "direction": "LONG",
        "curr_z": -3.14, "price_a": 418.20, "price_b": 175.40, "beta": 2.33,
        "shares": 50,    "is_cointegrated": False, "adf_pval": 0.09,
        "half_life": 21.4, "r_squared": 0.72, "lookback": 252,
        "edge_bps": 12,  "sharpe_est": 0.9,
    },
    {
        "a": "GS",   "b": "MS",    "direction": "LONG",
        "curr_z": -2.02, "price_a": 512.70, "price_b": 107.30, "beta": 4.72,
        "shares": 20,    "is_cointegrated": True,  "adf_pval": 0.035,
        "half_life": 6.8,  "r_squared": 0.83, "lookback": 63,
        "edge_bps": 41,  "sharpe_est": 2.4,
    },
]


def get_active_signals() -> list:
    """Swap this body with your live signal computation."""
    return MOCK_SIGNALS


# =============================================================================
# HELPERS
# =============================================================================

def score_signal(sig: dict) -> int:
    """Composite 0-5 signal confidence score."""
    score = 0
    if sig["is_cointegrated"]:            score += 2
    if sig["adf_pval"] < 0.05:           score += 1
    if abs(sig["curr_z"]) > 2.5:        score += 1
    if sig.get("r_squared", 0) > 0.80:  score += 1
    return min(score, 5)


def compute_legs(sig: dict) -> dict:
    """Dollar-neutral leg sizing."""
    shares_a   = sig["shares"]
    shares_b   = int(round(shares_a * sig["beta"]))
    notional_a = shares_a * sig["price_a"]
    notional_b = shares_b * sig["price_b"]
    return {
        "shares_a":   shares_a,
        "shares_b":   shares_b,
        "notional_a": notional_a,
        "notional_b": notional_b,
        "imbalance":  notional_a - notional_b,
    }


# =============================================================================
# CARD RENDERER
# Key rule: every f-string is ONE line. All dynamic values are
# pre-computed into plain variables before any HTML string is built.
# =============================================================================

def render_trade_card(sig: dict) -> str:

    # ── Colours & labels ─────────────────────────────────────
    is_long   = sig["direction"] == "LONG"
    accent    = "#00d4a0" if is_long else "#f56565"
    bg_accent = "rgba(0,212,160,0.06)" if is_long else "rgba(245,101,101,0.05)"
    dir_label = sig["direction"]
    z_meaning = "Spread undervalued — buy A / sell B" if is_long else "Spread overvalued — sell A / buy B"
    dir_badge_bg     = "rgba(0,212,160,0.15)"  if is_long else "rgba(245,101,101,0.15)"
    dir_badge_border = "rgba(0,212,160,0.30)"  if is_long else "rgba(245,101,101,0.30)"

    # ── Z-bar geometry ───────────────────────────────────────
    z_pct    = min(abs(sig["curr_z"]) / MAX_Z_VIZ, 1.0) * 50
    bar_left  = f"{50 - z_pct:.1f}%" if is_long else "50%"
    bar_width = f"{z_pct:.1f}%"
    z_str     = f"{sig['curr_z']:.2f}"

    # ── Execution legs ───────────────────────────────────────
    legs      = compute_legs(sig)
    leg1_verb = "BUY"  if is_long else "SELL"
    leg2_verb = "SELL" if is_long else "BUY"
    leg1_col  = "#00d4a0" if is_long else "#f56565"
    leg2_col  = "#f56565" if is_long else "#00d4a0"
    leg1_bg   = "rgba(0,212,160,0.12)"  if is_long else "rgba(245,101,101,0.10)"
    leg2_bg   = "rgba(245,101,101,0.10)" if is_long else "rgba(0,212,160,0.12)"
    imb_color = "#00d4a0" if abs(legs["imbalance"]) < 500 else "#f5a623"

    # Pre-compute every interpolated string (no f-strings inside HTML blocks)
    ticker_a      = sig["a"]
    ticker_b      = sig["b"]
    beta_str      = f"{sig['beta']:.2f}"
    shares_a_str  = str(legs["shares_a"])
    shares_b_str  = str(legs["shares_b"])
    price_a_str   = f"${sig['price_a']:.2f}"
    price_b_str   = f"${sig['price_b']:.2f}"
    notional_a_str = f"${legs['notional_a']:,.0f}"
    notional_b_str = f"${legs['notional_b']:,.0f}"
    imbalance_str  = f"${abs(legs['imbalance']):,.0f}"

    # ── Signal quality pips ──────────────────────────────────
    score  = score_signal(sig)
    pip_on = "#00d4a0" if (is_long and sig["is_cointegrated"]) else ("#f56565" if not is_long else "#f5a623")
    pip_parts = []
    for i in range(5):
        bg = pip_on if i < score else "#1e2330"
        pip_parts.append(
            '<div style="width:14px;height:4px;border-radius:1px;background:' + bg + ';"></div>'
        )
    pips = "".join(pip_parts)
    score_str = str(score)

    # ── Cointegration footer ─────────────────────────────────
    coint_color = "#00d4a0" if sig["is_cointegrated"] else "#f5a623"
    coint_text  = (
        "Cointegrated (p=" + f"{sig['adf_pval']:.3f}" + ")"
        if sig["is_cointegrated"]
        else "NOT cointegrated (p=" + f"{sig['adf_pval']:.2f}" + ")"
    )

    warn_banner = (
        ""
        if sig["is_cointegrated"]
        else (
            '<div style="background:rgba(245,166,23,0.07);border-top:1px solid rgba(245,166,23,0.2);'
            'padding:7px 16px;display:flex;align-items:center;gap:8px;">'
            '<span style="font-family:monospace;font-size:10px;font-weight:600;color:#f5a623;">!</span>'
            '<span style="font-family:monospace;font-size:10px;color:rgba(245,166,23,0.8);">'
            "Pair not cointegrated — reduce sizing, widen stop by 0.5σ"
            "</span></div>"
        )
    )

    # ── Metric cells ─────────────────────────────────────────
    hl     = sig.get("half_life", None)
    r2     = sig.get("r_squared", None)
    edge   = sig.get("edge_bps", None)
    sharpe = sig.get("sharpe_est", None)
    lkbk   = sig.get("lookback", 63)
    sharpe_color = "#00d4a0" if (sharpe and sharpe > 1.5) else "#f5a623"
    adf_color    = "#00d4a0" if sig["adf_pval"] < 0.05 else "#f5a623"

    hl_str     = (f"{hl:.1f}d" if isinstance(hl, float) else "–")
    r2_str     = (f"{r2 * 100:.0f}%" if r2 is not None else "–")
    edge_str   = (f"{edge}bps" if edge is not None else "–")
    sharpe_str = (f"{sharpe:.1f}" if sharpe is not None else "–")
    adf_str    = f"{sig['adf_pval']:.3f}"
    lkbk_str   = f"{lkbk}d"
    lkbk_full  = f"{lkbk}d lookback"
    entry_str  = f"Entry \u00b1{ENTRY_Z:.1f} / Exit \u00b1{EXIT_Z:.1f}"

    metric_rows = [
        ("Half-Life",  hl_str,     "#e8eaf0"),
        ("R\u00b2",   r2_str,     "#e8eaf0"),
        ("Edge",       edge_str,   "#e8c96d"),
        ("Sharpe Est", sharpe_str, sharpe_color),
        ("ADF p-val",  adf_str,    adf_color),
        ("Lookback",   lkbk_str,   "#e8eaf0"),
    ]

    metric_cells = []
    for lab, val, col in metric_rows:
        metric_cells.append(
            '<div style="background:#181c24;padding:8px 10px;border-radius:3px;">'
            '<p style="margin:0 0 3px;font-size:9px;color:#4a5568;font-family:monospace;'
            'text-transform:uppercase;letter-spacing:0.08em;">' + lab + "</p>"
            '<p style="margin:0;font-family:monospace;font-size:13px;font-weight:500;color:' + col + ';">' + val + "</p>"
            "</div>"
        )
    metrics_html = (
        '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:14px;">'
        + "".join(metric_cells)
        + "</div>"
    )

    # ── Execution block ──────────────────────────────────────
    exec_html = (
        '<div style="background:#181c24;border-radius:3px;padding:10px 12px;margin-bottom:12px;">'
        '<p style="margin:0 0 8px;font-size:9px;color:#4a5568;font-family:monospace;'
        'text-transform:uppercase;letter-spacing:0.1em;">Execution (Dollar-Neutral)</p>'

        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">'
        '<span style="font-family:monospace;font-size:11px;font-weight:600;color:' + leg1_col + ';background:' + leg1_bg + ';padding:2px 7px;border-radius:2px;">' + leg1_verb + "</span>"
        '<span style="font-family:monospace;font-size:12px;color:#e8eaf0;padding:0 8px;">' + ticker_a + "</span>"
        '<span style="font-family:monospace;font-size:11px;color:#8892a4;">' + shares_a_str + " shs @ " + price_a_str + "</span>"
        '<span style="font-family:monospace;font-size:11px;color:#4a5568;">' + notional_a_str + "</span>"
        "</div>"

        '<div style="display:flex;justify-content:space-between;align-items:center;">'
        '<span style="font-family:monospace;font-size:11px;font-weight:600;color:' + leg2_col + ';background:' + leg2_bg + ';padding:2px 7px;border-radius:2px;">' + leg2_verb + "</span>"
        '<span style="font-family:monospace;font-size:12px;color:#e8eaf0;padding:0 8px;">' + ticker_b + "</span>"
        '<span style="font-family:monospace;font-size:11px;color:#8892a4;">' + shares_b_str + " shs @ " + price_b_str + "</span>"
        '<span style="font-family:monospace;font-size:11px;color:#4a5568;">' + notional_b_str + "</span>"
        "</div>"

        '<p style="margin:8px 0 0;font-size:9px;color:#4a5568;font-family:monospace;text-align:right;">'
        'Net imbalance: <span style="color:' + imb_color + ';">' + imbalance_str + "</span></p>"
        "</div>"
    )

    # ── Assemble card  (only simple {var} references here) ──
    return f"""
<div style="background:#111318;border-radius:4px;overflow:hidden;margin-bottom:12px;
            border:1px solid rgba(255,255,255,0.07);border-top:2px solid {accent};">

  <div style="padding:14px 16px 12px;border-bottom:1px solid rgba(255,255,255,0.07);
              display:flex;justify-content:space-between;align-items:flex-start;
              background:{bg_accent};">
    <div>
      <p style="margin:0;font-family:monospace;font-size:15px;font-weight:600;
                letter-spacing:0.04em;color:{accent};">
        {ticker_a} <span style="color:#4a5568;font-weight:300;">/</span> {ticker_b}
      </p>
      <p style="margin:3px 0 0;font-size:10px;color:#4a5568;font-family:monospace;">
        &beta;={beta_str} &middot; {lkbk_full}
      </p>
    </div>
    <span style="font-family:monospace;font-size:9px;font-weight:600;letter-spacing:0.14em;
          padding:3px 8px;border-radius:2px;color:{accent};
          background:{dir_badge_bg};border:1px solid {dir_badge_border};">
      {dir_label}
    </span>
  </div>

  <div style="padding:14px 16px;">

    <div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;">
      <div>
        <p style="margin:0 0 3px;font-size:9px;color:#4a5568;font-family:monospace;
                  text-transform:uppercase;letter-spacing:0.08em;">Z-Score</p>
        <p style="margin:0;font-family:monospace;font-size:26px;font-weight:600;
                  line-height:1;color:{accent};">{z_str}</p>
      </div>
      <div style="flex:1;">
        <p style="margin:0 0 4px;font-size:10px;color:#8892a4;font-family:monospace;">
          {entry_str}
        </p>
        <div style="height:4px;background:#1e2330;border-radius:2px;position:relative;">
          <div style="position:absolute;height:100%;border-radius:2px;
                      left:{bar_left};width:{bar_width};background:{accent};"></div>
          <div style="position:absolute;left:50%;top:-2px;width:1px;height:8px;
                      background:rgba(255,255,255,0.13);"></div>
        </div>
        <p style="margin:4px 0 0;font-size:10px;color:#4a5568;font-family:monospace;">
          {z_meaning}
        </p>
      </div>
    </div>

    {metrics_html}
    {exec_html}

  </div>

  <div style="padding:10px 16px;border-top:1px solid rgba(255,255,255,0.07);
              display:flex;justify-content:space-between;align-items:center;">
    <div style="display:flex;align-items:center;gap:8px;">
      <span style="font-size:9px;color:#4a5568;font-family:monospace;
            text-transform:uppercase;letter-spacing:0.08em;">Signal Quality</span>
      <div style="display:flex;gap:3px;">{pips}</div>
      <span style="font-family:monospace;font-size:10px;color:#4a5568;">{score_str}/5</span>
    </div>
    <div style="display:flex;align-items:center;gap:5px;">
      <div style="width:5px;height:5px;border-radius:50%;background:{coint_color};"></div>
      <span style="font-family:monospace;font-size:10px;color:{coint_color};">{coint_text}</span>
    </div>
  </div>

  {warn_banner}
</div>"""


# =============================================================================
# MAIN DASHBOARD SECTION  —  call this from your app.py
# =============================================================================

def render_active_trade_summary(active_signals: list) -> None:
    """
    Drop-in replacement for the original st.subheader + card loop.
    Call anywhere in your app:
        render_active_trade_summary(get_active_signals())
    """
    now_str = datetime.now().strftime("%H:%M:%S") + " ET"

    st.markdown(
        '<div style="display:flex;align-items:center;justify-content:space-between;'
        'padding-bottom:12px;border-bottom:1px solid rgba(255,255,255,0.07);margin-bottom:16px;">'
        '<p style="margin:0;font-family:monospace;font-size:11px;letter-spacing:0.12em;'
        'color:#8892a4;text-transform:uppercase;">'
        '<span style="display:inline-block;width:6px;height:6px;border-radius:50%;'
        'background:#00d4a0;margin-right:8px;vertical-align:middle;'
        'animation:none;"></span>'
        "Active Trade Summary &amp; Execution Points</p>"
        '<p style="margin:0;font-family:monospace;font-size:10px;color:#4a5568;">'
        + now_str + " &middot; " + str(len(active_signals)) + " signal"
        + ("s" if len(active_signals) != 1 else "") + " active</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    if not active_signals:
        st.markdown(
            '<div style="text-align:center;padding:48px 24px;background:#111318;'
            'border-radius:4px;border:1px dashed rgba(255,255,255,0.13);">'
            '<p style="font-family:monospace;font-size:12px;color:#8892a4;letter-spacing:0.08em;">'
            "NO ACTIVE SIGNALS</p>"
            '<p style="font-size:11px;color:#4a5568;font-family:monospace;margin-top:6px;">'
            f"All Z-scores within \u00b1{ENTRY_Z:.1f}\u03c3 \u2014 monitoring for deviations</p>"
            "</div>",
            unsafe_allow_html=True,
        )
        return

    # Sort: highest |z| first; non-cointegrated pairs pushed down
    sorted_sigs = sorted(
        active_signals,
        key=lambda s: (not s["is_cointegrated"], -abs(s["curr_z"]))
    )

    # Two-column grid
    col_count = min(len(sorted_sigs), 2)
    cols = st.columns(col_count)
    for i, sig in enumerate(sorted_sigs):
        with cols[i % col_count]:
            st.markdown(render_trade_card(sig), unsafe_allow_html=True)


# =============================================================================
# STREAMLIT ENTRY POINT
# =============================================================================

def main():
    st.set_page_config(
        page_title="Omni-Arb | Trade Dashboard",
        page_icon="⚡",
        layout="wide",
    )

    # Dark background to match the card aesthetic
    st.markdown(
        "<style>body, .stApp { background-color: #0a0c10; } "
        "section[data-testid='stSidebar'] { background-color: #0e1117; }</style>",
        unsafe_allow_html=True,
    )

    st.title("Omni-Arb  |  Stat-Arb Monitor")

    # ── Pull signals (swap mock for live) ──────────────────
    active_signals = get_active_signals()

    # ── Section 3: Active Trade Summary ───────────────────
    render_active_trade_summary(active_signals)


if __name__ == "__main__":
    main()

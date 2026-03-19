def render_trade_card(sig: dict) -> str:
    is_long       = sig["direction"] == "LONG"
    accent        = "#00d4a0" if is_long else "#f56565"
    bg_accent     = "rgba(0,212,160,0.06)" if is_long else "rgba(245,101,101,0.05)"
    dir_label     = sig["direction"]
    z_meaning     = "Spread undervalued — buy A / sell B" if is_long else "Spread overvalued — sell A / buy B"

    # Z-bar
    z_pct     = min(abs(sig["curr_z"]) / MAX_Z_VIZ, 1.0) * 50
    bar_left  = f"{50 - z_pct:.1f}%" if is_long else "50%"
    bar_width = f"{z_pct:.1f}%"

    # Leg sizing
    legs      = compute_legs(sig)
    leg1_verb, leg2_verb = ("BUY", "SELL") if is_long else ("SELL", "BUY")
    leg1_col  = "#00d4a0" if is_long else "#f56565"
    leg2_col  = "#f56565" if is_long else "#00d4a0"
    leg1_bg   = "rgba(0,212,160,0.12)"  if is_long else "rgba(245,101,101,0.10)"
    leg2_bg   = "rgba(245,101,101,0.10)" if is_long else "rgba(0,212,160,0.12)"
    imb_color = "#00d4a0" if abs(legs["imbalance"]) < 500 else "#f5a623"

    # Signal quality
    score   = score_signal(sig)
    pip_on  = "#00d4a0" if (is_long and sig["is_cointegrated"]) else ("#f56565" if not is_long else "#f5a623")
    pips    = "".join(
        f'<div style="width:14px;height:4px;border-radius:1px;'
        f'background:{"" + pip_on if i < score else "#1e2330"};"></div>'
        for i in range(5)
    )

    # Cointegration
    coint_color = "#00d4a0" if sig["is_cointegrated"] else "#f5a623"
    coint_text  = (
        f"Cointegrated (p={sig['adf_pval']:.3f})"
        if sig["is_cointegrated"]
        else f"NOT cointegrated (p={sig['adf_pval']:.2f})"
    )

    warn_banner = "" if sig["is_cointegrated"] else (
        '<div style="background:rgba(245,166,23,0.07);border-top:1px solid rgba(245,166,23,0.2);'
        'padding:7px 16px;display:flex;align-items:center;gap:8px;">'
        '<span style="font-family:monospace;font-size:10px;font-weight:600;color:#f5a623;">!</span>'
        '<span style="font-family:monospace;font-size:10px;color:rgba(245,166,23,0.8);">'
        "Pair not cointegrated — reduce sizing, widen stop by 0.5σ"
        "</span></div>"
    )

    # Shorthand values
    hl     = sig.get("half_life", "–")
    r2     = sig.get("r_squared", None)
    edge   = sig.get("edge_bps", "–")
    sharpe = sig.get("sharpe_est", None)
    lkbk   = sig.get("lookback", 63)
    sharpe_color = "#00d4a0" if (sharpe and sharpe > 1.5) else "#f5a623"

    # ── PRE-BUILD all sub-HTML before the main f-string ──────
    metric_rows = [
        ("Half-Life", f"{hl:.1f}d" if isinstance(hl, float) else str(hl), "#e8eaf0"),
        ("R²",        f"{r2 * 100:.0f}%" if r2 is not None else "–",       "#e8eaf0"),
        ("Edge",      f"{edge}bps",                                          "#e8c96d"),
        ("Sharpe Est",f"{sharpe:.1f}" if sharpe is not None else "–",       sharpe_color),
        ("ADF p-val", f"{sig['adf_pval']:.3f}",                             "#00d4a0" if sig["adf_pval"] < 0.05 else "#f5a623"),
        ("Lookback",  f"{lkbk}d",                                            "#e8eaf0"),
    ]

    metrics_html = (
        '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:14px;">'
        + "".join(
            '<div style="background:#181c24;padding:8px 10px;border-radius:3px;">'
            f'<p style="margin:0 0 3px;font-size:9px;color:#4a5568;font-family:monospace;'
            f'text-transform:uppercase;letter-spacing:0.08em;">{lab}</p>'
            f'<p style="margin:0;font-family:monospace;font-size:13px;font-weight:500;color:{col};">{val}</p>'
            "</div>"
            for lab, val, col in metric_rows
        )
        + "</div>"
    )

    exec_html = (
        '<div style="background:#181c24;border-radius:3px;padding:10px 12px;margin-bottom:12px;">'
        '<p style="margin:0 0 8px;font-size:9px;color:#4a5568;font-family:monospace;'
        'text-transform:uppercase;letter-spacing:0.1em;">Execution (Dollar-Neutral)</p>'
        # Leg A
        '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;">'
        f'<span style="font-family:monospace;font-size:11px;font-weight:600;color:{leg1_col};'
        f'background:{leg1_bg};padding:2px 7px;border-radius:2px;">{leg1_verb}</span>'
        f'<span style="font-family:monospace;font-size:12px;color:#e8eaf0;padding:0 8px;">{sig["a"]}</span>'
        f'<span style="font-family:monospace;font-size:11px;color:#8892a4;">{legs["shares_a"]} shs @ ${sig["price_a"]:.2f}</span>'
        f'<span style="font-family:monospace;font-size:11px;color:#4a5568;">${legs["notional_a"]:,.0f}</span>'
        "</div>"
        # Leg B
        '<div style="display:flex;justify-content:space-between;align-items:center;">'
        f'<span style="font-family:monospace;font-size:11px;font-weight:600;color:{leg2_col};'
        f'background:{leg2_bg};padding:2px 7px;border-radius:2px;">{leg2_verb}</span>'
        f'<span style="font-family:monospace;font-size:12px;color:#e8eaf0;padding:0 8px;">{sig["b"]}</span>'
        f'<span style="font-family:monospace;font-size:11px;color:#8892a4;"

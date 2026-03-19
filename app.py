def compute_legs(sig, capital=100000):
    """
    Hedge Fund Logic: Calculates dollar-neutral position sizing.
    Assumes a $100,000 'unit' per trade.
    """
    # Allocate 50% of capital to the primary leg (A)
    notional_a = capital / 2
    shares_a = int(notional_a / sig["price_a"])
    actual_notional_a = shares_a * sig["price_a"]
    
    # Calculate Leg B based on the Beta (Hedge Ratio)
    # Hedge Ratio = Beta * (Price_A / Price_B)
    shares_b = int(shares_a * sig["beta"])
    actual_notional_b = shares_b * sig["price_b"]
    
    imbalance = actual_notional_a - actual_notional_b
    
    return {
        "shares_a": shares_a,
        "notional_a": actual_notional_a,
        "shares_b": shares_b,
        "notional_b": actual_notional_b,
        "imbalance": imbalance
    }

def render_trade_card(sig: dict) -> str:
    is_long = sig["direction"] == "LONG"
    accent = "#00d4a0" if is_long else "#f56565"
    bg_accent = "rgba(0,212,160,0.06)" if is_long else "rgba(245,101,101,0.05)"
    
    # Meaning of the Z-score for the user
    z_desc = f"{'Undervalued' if is_long else 'Overvalued'} (Deviation: {abs(sig['curr_z']):.2f}σ)"
    z_action = "BUY A / SELL B" if is_long else "SELL A / BUY B"

    # Sizing
    legs = compute_legs(sig)
    leg1_verb, leg2_verb = ("BUY", "SELL") if is_long else ("SELL", "BUY")
    
    # Constructing the HTML in parts to avoid f-string nesting limits
    html = f"""
    <div style="background:{bg_accent}; border: 1px solid {accent}; padding: 20px; border-radius: 8px; margin-bottom: 20px; font-family: 'Inter', sans-serif;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:15px;">
            <div>
                <h3 style="margin:0; color:{accent}; font-size:18px;">{sig['a']} / {sig['b']} — {sig['direction']}</h3>
                <p style="margin:0; font-size:12px; color:#8892a4;">{z_desc} | {z_action}</p>
            </div>
            <div style="text-align:right;">
                <span style="font-family:monospace; font-size:20px; font-weight:bold; color:{accent};">Z: {sig['curr_z']:.2f}</span>
            </div>
        </div>

        <div style="background:rgba(0,0,0,0.2); padding:12px; border-radius:4px; margin-bottom:12px;">
            <p style="margin:0 0 8px 0; font-size:10px; color:#4a5568; text-transform:uppercase; letter-spacing:1px; font-weight:bold;">Execution Plan (Dollar-Neutral)</p>
            
            <div style="display:flex; justify-content:space-between; margin-bottom:6px; font-family:monospace; font-size:13px;">
                <span style="color:{accent}; font-weight:bold;">{leg1_verb} {sig['a']}</span>
                <span style="color:#e8eaf0;">{legs['shares_a']} shares @ ${sig['price_a']:.2f}</span>
                <span style="color:#4a5568;">${legs['notional_a']:,.0f}</span>
            </div>

            <div style="display:flex; justify-content:space-between; font-family:monospace; font-size:13px;">
                <span style="color:{'#f56565' if is_long else '#00d4a0'}; font-weight:bold;">{leg2_verb} {sig['b']}</span>
                <span style="color:#e8eaf0;">{legs['shares_b']} shares @ ${sig['price_b']:.2f}</span>
                <span style="color:#4a5568;">${legs['notional_b']:,.0f}</span>
            </div>
        </div>
        
        <div style="font-size:11px; color:#4a5568; display:flex; justify-content:space-between;">
            <span>Cointegration: {'PASS' if sig['is_cointegrated'] else 'FAIL'} (p={sig['adf_pval']:.3f})</span>
            <span>Est. Imbalance: ${abs(legs['imbalance']):,.2f}</span>
        </div>
    </div>
    """
    return html

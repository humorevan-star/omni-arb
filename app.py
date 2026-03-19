# ... (Keep previous imports and get_data() function)

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
            st.subheader(f"📊 {a} vs {b} Analysis")
            strat_name, strat_desc, dte = get_instrument_type(pair_vol)

            # === DYNAMIC TICKER MAPPING ===
            if abs(curr_z) > ENTRY_Z:
                if curr_z < -ENTRY_Z:
                    # Spread is low -> Buy the first stock (A), Sell the second (B)
                    color_theme, border_color = "#d4edda", "#28a745"
                    action_title = f"LONG SPREAD: BUY {a}"
                    primary_trade = f"🟢 BUY {a} / 🔴 SELL {b}"
                    hedge_ticker = a
                else:
                    # Spread is high -> Sell the first stock (A), Buy the second (B)
                    color_theme, border_color = "#f8d7da", "#dc3545"
                    action_title = f"SHORT SPREAD: SELL {a}"
                    primary_trade = f"🔴 SELL {a} / 🟢 BUY {b}"
                    hedge_ticker = a

                # === FIXED SIGNAL BANNER WITH TICKERS ===
                st.markdown(f"""
                    <div style="background-color:{color_theme}; padding:20px; border-radius:10px; border: 3px solid {border_color}; color:black;">
                        <h2 style="margin:0; font-size:24px;">🔥 SIGNAL: {action_title}</h2>
                        <hr style="border-top: 1px solid {border_color}; margin: 10px 0;">
                        <p style="font-size:20px; margin-bottom:5px;"><b>EXECUTION:</b> {primary_trade}</p>
                        <p style="font-size:16px; margin:0;"><b>RATIO:</b> 1 share of {a} per {round(beta, 2)} shares of {b}</p>
                        <p style="font-size:16px; margin-top:10px;"><b>HEDGE:</b> {strat_name} on <b>{hedge_ticker}</b> ({dte})</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info(f"⚪ **NO TRADE** for {a}/{b} – Z-Score: {curr_z:.2f}")

            # ... (Keep Plotly chart code below)

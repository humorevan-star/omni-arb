import numpy as np

def generate_trade_orders(s1_ticker, s2_ticker, z_score, s1_price, s2_price, pair_volatility, account_val=1000):
    
    # Check if a trade is triggered
    if abs(z_score) < 2.25:
        return "Z-Score neutral. Monitoring..."

    # 1. Determine Direction
    if z_score < -2.25:
        # Long Spread: S1 is Undervalued (Buy), S2 is Overvalued (Sell)
        s1_sentiment = "BULLISH"
        s2_sentiment = "BEARISH"
    elif z_score > 2.25:
        # Short Spread: S1 is Overvalued (Sell), S2 is Undervalued (Buy)
        s1_sentiment = "BEARISH"
        s2_sentiment = "BULLISH"

    # 2. Determine Strategy based on Volatility Context
    print(f"\n--- NEW SIGNAL: {s1_ticker} / {s2_ticker} ---")
    print(f"Z-Score: {z_score:.2f} | Pair Volatility: {pair_volatility:.1%}")
    
    # ALLOCATION: Risk ~20% of account per trade leg ($100 per leg for a $1k account)
    # 1 option contract = 100 shares. 
    qty = 1 # Keeping it safe for the $1,000 cash account level

    if pair_volatility > 0.25:
        strategy_name = "RATIO BACKSPREAD (1:2)"
        print(f"Strategy Selected: {strategy_name} (High Volatility Environment)")
        
        # S1 Orders
        if s1_sentiment == "BULLISH":
            print(f"[{s1_ticker} Leg]: SELL {qty} ATM Call, BUY {qty * 2} OTM Calls")
        else:
            print(f"[{s1_ticker} Leg]: SELL {qty} ATM Put, BUY {qty * 2} OTM Puts")
            
        # S2 Orders
        if s2_sentiment == "BULLISH":
            print(f"[{s2_ticker} Leg]: SELL {qty} ATM Call, BUY {qty * 2} OTM Calls")
        else:
            print(f"[{s2_ticker} Leg]: SELL {qty} ATM Put, BUY {qty * 2} OTM Puts")

    elif 0.12 <= pair_volatility <= 0.25:
        strategy_name = "VERTICAL SPREAD"
        print(f"Strategy Selected: {strategy_name} (Medium Volatility Environment)")
        
        # S1 Orders
        if s1_sentiment == "BULLISH":
            print(f"[{s1_ticker} Leg]: BUY {qty} ATM Call, SELL {qty} OTM Call (Bull Call Spread)")
        else:
            print(f"[{s1_ticker} Leg]: BUY {qty} ATM Put, SELL {qty} OTM Put (Bear Put Spread)")
            
        # S2 Orders
        if s2_sentiment == "BULLISH":
            print(f"[{s2_ticker} Leg]: BUY {qty} ATM Call, SELL {qty} OTM Call (Bull Call Spread)")
        else:
            print(f"[{s2_ticker} Leg]: BUY {qty} ATM Put, SELL {qty} OTM Put (Bear Put Spread)")

    else:
        strategy_name = "PURE STOCKS"
        print(f"Strategy Selected: {strategy_name} (Low Volatility Environment - Avoiding Theta Decay)")
        
        # Determine affordable fractional/whole shares based on $250 max allocation
        s1_shares = round(250 / s1_price, 2)
        s2_shares = round(250 / s2_price, 2)
        
        if s1_sentiment == "BULLISH":
            print(f"[{s1_ticker} Leg]: BUY {s1_shares} Shares")
            print(f"[{s2_ticker} Leg]: SHORT {s2_shares} Shares")
        else:
            print(f"[{s1_ticker} Leg]: SHORT {s1_shares} Shares")
            print(f"[{s2_ticker} Leg]: BUY {s2_shares} Shares")

    print("---------------------------------------")

# Example Triggers:
generate_trade_orders("NVDA", "AMD", 2.60, 178.44, 198.70, 0.35)  # High Volatility Scenario
generate_trade_orders("V", "MA", -2.45, 297.78, 489.25, 0.15)     # Medium Volatility Scenario
generate_trade_orders("PEP", "KO", 2.30, 153.57, 75.77, 0.08)     # Low Volatility Scenario

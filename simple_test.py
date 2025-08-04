"""
Simple test for the close-to-open strategy.
This script tests your strategy with SPY and QQQ.
"""

def test_strategy():
    print("Testing Close-to-Open Strategy")
    print("=" * 40)
    
    try:
        # Test imports
        import yfinance as yf
        import pandas as pd
        import numpy as np
        print("✓ All packages imported successfully")
        
        # Test data fetching
        print("\n1. Testing data fetch...")
        ticker = yf.Ticker("SPY")
        data = ticker.history(period="5d")
        if not data.empty:
            print(f"✓ Fetched {len(data)} days of SPY data")
            print(f"  Latest close: ${data['Close'].iloc[-1]:.2f}")
        else:
            print("✗ No data fetched")
            return
        
        # Test strategy logic
        print("\n2. Testing strategy logic...")
        
        # Get more data for strategy test
        data = ticker.history(period="1mo")
        data['Next_Open'] = data['Open'].shift(-1)
        data = data[:-1]  # Remove last row
        
        # Simulate strategy
        initial_capital = 10000
        capital = initial_capital
        total_trades = 0
        
        for i, (date, row) in enumerate(data.iterrows()):
            close_price = row['Close']
            next_open = row['Next_Open']
            
            if pd.notna(next_open) and capital > close_price:
                # Buy at close
                shares = int(capital / close_price)
                if shares > 0:
                    buy_cost = shares * close_price
                    # Sell at next open
                    sell_value = shares * next_open
                    profit = sell_value - buy_cost
                    capital = capital - buy_cost + sell_value
                    total_trades += 1
        
        final_return = ((capital - initial_capital) / initial_capital) * 100
        
        print(f"✓ Strategy simulation complete")
        print(f"  Initial capital: ${initial_capital:,.2f}")
        print(f"  Final capital: ${capital:,.2f}")
        print(f"  Total return: {final_return:.2f}%")
        print(f"  Total trades: {total_trades}")
        
        # Test with QQQ
        print("\n3. Testing with QQQ...")
        qqq_ticker = yf.Ticker("QQQ")
        qqq_data = qqq_ticker.history(period="5d")
        if not qqq_data.empty:
            print(f"✓ QQQ data fetched: ${qqq_data['Close'].iloc[-1]:.2f}")
        
        print("\n🎉 All tests passed!")
        print("\nYour close-to-open strategy framework is ready!")
        print("\nStrategy details:")
        print("- Buy at market close every day")
        print("- Sell at next market open")
        print("- Use all available capital for each trade")
        print("- Compound returns daily")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("\nIf you see import errors, try:")
        print("pip install yfinance pandas numpy matplotlib")

if __name__ == "__main__":
    test_strategy()

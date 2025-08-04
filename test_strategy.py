"""
Quick test script for the close-to-open strategy.
Run this after installing dependencies: pip install yfinance pandas numpy matplotlib

Usage:
python test_strategy.py
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test if all required packages are available."""
    try:
        import pandas as pd
        import numpy as np
        import yfinance as yf
        print("✓ All required packages are available")
        return True
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        print("Please install missing packages with:")
        print("pip install yfinance pandas numpy matplotlib")
        return False

def test_data_fetch():
    """Test fetching data from Yahoo Finance."""
    try:
        from src.data_fetcher import StockDataFetcher
        
        fetcher = StockDataFetcher()
        print("\n--- Testing Data Fetching ---")
        
        # Test fetching SPY data
        data = fetcher.get_stock_data("SPY", period="5d")
        
        if not data.empty:
            print(f"✓ Successfully fetched SPY data: {len(data)} days")
            print(f"  Latest close price: ${data['Close'].iloc[-1]:.2f}")
            return True
        else:
            print("✗ No data received")
            return False
            
    except Exception as e:
        print(f"✗ Error fetching data: {e}")
        return False

def test_backtesting():
    """Test the backtesting functionality."""
    try:
        from src.close_to_open_strategy import CloseToOpenBacktester
        
        print("\n--- Testing Backtesting ---")
        backtester = CloseToOpenBacktester(initial_capital=10000)
        
        # Test with a small dataset first
        results = backtester.backtest_close_to_open("SPY", period="1mo")
        
        if 'error' in results:
            print(f"✗ Backtesting error: {results['error']}")
            return False
        
        print("✓ Backtesting successful!")
        print(f"  Strategy: {results['strategy']}")
        print(f"  Return: {results['return_percentage']:.2f}%")
        print(f"  Total trades: {results['total_trades']}")
        
        return True
        
    except Exception as e:
        print(f"✗ Backtesting error: {e}")
        return False

def run_strategy_demo():
    """Run a quick demo of the strategy."""
    try:
        from src.close_to_open_strategy import CloseToOpenBacktester
        
        print("\n" + "="*50)
        print("CLOSE-TO-OPEN STRATEGY DEMO")
        print("="*50)
        
        backtester = CloseToOpenBacktester(initial_capital=10000)
        symbols = ["SPY", "QQQ"]
        
        for symbol in symbols:
            print(f"\n--- {symbol} Strategy Test (Last 3 months) ---")
            
            results = backtester.backtest_close_to_open(symbol, period="3mo")
            
            if 'error' in results:
                print(f"Error: {results['error']}")
                continue
            
            print(f"Initial Capital: ${results['initial_capital']:,.2f}")
            print(f"Final Value: ${results['final_value']:,.2f}")
            print(f"Strategy Return: {results['return_percentage']:.2f}%")
            print(f"Buy & Hold Return: {results['buy_hold_return']:.2f}%")
            print(f"Excess Return: {results['excess_return']:.2f}%")
            print(f"Win Rate: {results['win_rate']:.1f}%")
            print(f"Total Trade Pairs: {results['total_trade_pairs']}")
            
            if results['excess_return'] > 0:
                print(f"🎉 Strategy outperformed buy & hold by {results['excess_return']:.2f}%")
            else:
                print(f"📉 Strategy underperformed buy & hold by {abs(results['excess_return']):.2f}%")
        
        print(f"\n" + "="*50)
        print("Demo complete! 🚀")
        
    except Exception as e:
        print(f"Demo error: {e}")

if __name__ == "__main__":
    print("Stock Investor - Strategy Testing")
    print("=================================")
    
    # Test imports
    if not test_imports():
        sys.exit(1)
    
    # Test data fetching
    if not test_data_fetch():
        print("Data fetching failed. Check your internet connection.")
        sys.exit(1)
    
    # Test backtesting
    if not test_backtesting():
        print("Backtesting failed.")
        sys.exit(1)
    
    # Run demo
    run_strategy_demo()
    
    print("\n✅ All tests passed! You can now use the full application.")
    print("\nTry these commands:")
    print("python -m src.main demo-strategy")
    print("python -c \"from src.close_to_open_strategy import CloseToOpenBacktester; b=CloseToOpenBacktester(); print('Strategy ready!')\"")
    print("# Or create a simple test:")
    print("# from src.close_to_open_strategy import CloseToOpenBacktester")
    print("# backtester = CloseToOpenBacktester(10000)")
    print("# results = backtester.backtest_close_to_open('SPY', period='1mo')")
    print("# print(f'Return: {results[\"return_percentage\"]:.2f}%')")

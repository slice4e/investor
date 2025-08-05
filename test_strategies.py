"""
Test Script for Buy and Hold Strategy

This script tests the buy and hold strategy implementation to ensure
it works correctly with the data downloader integration.
"""

import sys
import os
from pathlib import Path
import logging

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_buy_and_hold_strategy():
    """Test the Buy and Hold strategy with sample data."""
    
    print("🧪 Testing Buy and Hold Strategy")
    print("=" * 40)
    
    try:
        # Import strategy manager which handles the imports correctly
        from backtesting.strategy_manager import StrategyManager
        manager = StrategyManager()
        
        # Use the manager to create the strategy
        from strategies.buy_and_hold.buy_and_hold_strategy import BuyAndHoldStrategy
        
        # Initialize strategy
        strategy = BuyAndHoldStrategy(
            initial_capital=10000.0,
            commission=0.0,
            sell_at_end=False
        )
        
        # Test parameters - using a well-known stock
        ticker = "AAPL"
        start_date = "2022-01-01"
        end_date = "2023-12-31"
        
        print(f"📊 Testing with {ticker} from {start_date} to {end_date}")
        print(f"💰 Initial capital: ${strategy.initial_capital:,.2f}")
        
        # Execute strategy
        result = strategy.execute_strategy(ticker, start_date, end_date)
        
        # Display results
        print(f"\n✅ Strategy executed successfully!")
        print(f"📈 Final portfolio value: ${result['final_value']:,.2f}")
        print(f"📊 Total return: {result['performance']['total_return_pct']:.2f}%")
        print(f"📅 Investment period: {result['start_date']} to {result['end_date']}")
        print(f"🔄 Total trades: {result['performance']['total_trades']}")
        
        if result['buy_price']:
            print(f"💵 Buy price: ${result['buy_price']:.2f}")
        if result['sell_price']:
            print(f"💰 Final price: ${result['sell_price']:.2f}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you're running from the project root directory")
        return False
        
    except Exception as e:
        print(f"❌ Strategy execution failed: {e}")
        logging.error("Strategy test failed", exc_info=True)
        return False

def test_strategy_manager():
    """Test the Strategy Manager functionality."""
    
    print("\n🧪 Testing Strategy Manager")
    print("=" * 30)
    
    try:
        from backtesting.strategy_manager import StrategyManager
        
        # Initialize manager
        manager = StrategyManager()
        
        print(f"📋 Available strategies: {manager.get_available_strategies()}")
        
        # Test running a strategy
        result = manager.run_strategy(
            strategy_name='buy_and_hold',
            ticker='MSFT',
            start_date='2022-06-01',
            end_date='2023-06-01',
            strategy_params={'initial_capital': 5000.0}
        )
        
        print(f"✅ Strategy Manager test successful!")
        print(f"📊 Result ticker: {result['ticker']}")
        print(f"💰 Final value: ${result['final_value']:,.2f}")
        
        # Test comparison
        comparison = manager.compare_strategies()
        print(f"\n📈 Comparison table generated with {len(comparison)} results")
        
        return True
        
    except Exception as e:
        print(f"❌ Strategy Manager test failed: {e}")
        logging.error("Strategy Manager test failed", exc_info=True)
        return False

if __name__ == "__main__":
    print("🚀 Investment Strategy Testing")
    print("=" * 50)
    
    # Test individual strategy
    strategy_test = test_buy_and_hold_strategy()
    
    # Test strategy manager
    manager_test = test_strategy_manager()
    
    # Summary
    print(f"\n📋 Test Summary")
    print("=" * 20)
    print(f"Buy and Hold Strategy: {'✅ PASS' if strategy_test else '❌ FAIL'}")
    print(f"Strategy Manager: {'✅ PASS' if manager_test else '❌ FAIL'}")
    
    if strategy_test and manager_test:
        print(f"\n🎉 All tests passed! The backtesting infrastructure is ready to use.")
    else:
        print(f"\n⚠️ Some tests failed. Please check the error messages above.")

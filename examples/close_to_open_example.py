"""
Example script for testing the close-to-open strategy on SPY and QQQ.

This script demonstrates how to use the backtesting framework to test
the intraday strategy: buy at close, sell at next open.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtester import Backtester, CloseToOpenStrategy
import pandas as pd


def test_close_to_open_strategy():
    """Test the close-to-open strategy on SPY and QQQ."""
    
    print("=" * 60)
    print("Close-to-Open Strategy Backtest")
    print("Strategy: Buy at close, sell at next open (every day)")
    print("=" * 60)
    
    # Initialize backtester
    backtester = Backtester(initial_capital=10000.0)
    
    # Test symbols
    symbols = ['SPY', 'QQQ']
    
    for symbol in symbols:
        print(f"\n--- Testing {symbol} ---")
        
        try:
            # Run backtest for close-to-open strategy
            results = backtester.backtest_close_to_open(
                symbol=symbol,
                period="1y"  # Last 1 year
            )
            
            if 'error' in results:
                print(f"Error: {results['error']}")
                continue
            
            # Display results
            print(f"Symbol: {results['symbol']}")
            print(f"Strategy: {results['strategy']}")
            print(f"Period: {results['period']}")
            print(f"Initial Capital: ${results['initial_capital']:,.2f}")
            print(f"Final Value: ${results['final_value']:,.2f}")
            print(f"Total Return: ${results['total_return']:,.2f}")
            print(f"Return %: {results['return_percentage']:.2f}%")
            print(f"Buy & Hold Return: {results['buy_hold_return']:.2f}%")
            print(f"Excess Return: {results['excess_return']:.2f}%")
            print(f"Volatility: {results['volatility']:.2f}")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
            print(f"Win Rate: {results['win_rate']:.1f}%")
            print(f"Total Trades: {results['total_trades']}")
            print(f"Trade Pairs: {results['total_trade_pairs']}")
            
            # Show last few trades
            if results['trades']:
                print(f"\nLast 5 trades:")
                recent_trades = results['trades'][-10:]  # Last 10 transactions (5 pairs)
                for trade in recent_trades:
                    print(f"  {trade['date'].strftime('%Y-%m-%d')}: {trade['type']} {trade['shares']} shares at ${trade['price']:.2f}")
            
        except Exception as e:
            print(f"Error testing {symbol}: {str(e)}")
    
    print("\n" + "=" * 60)


def compare_symbols():
    """Compare the strategy performance across different symbols."""
    
    print("\n" + "=" * 60)
    print("Strategy Comparison Across Symbols")
    print("=" * 60)
    
    backtester = Backtester(initial_capital=10000.0)
    symbols = ['SPY', 'QQQ', 'IWM', 'DIA']  # Different ETFs
    
    results_summary = []
    
    for symbol in symbols:
        try:
            results = backtester.backtest_close_to_open(symbol=symbol, period="1y")
            
            if 'error' not in results:
                results_summary.append({
                    'Symbol': symbol,
                    'Return %': results['return_percentage'],
                    'Buy & Hold %': results['buy_hold_return'],
                    'Excess Return %': results['excess_return'],
                    'Sharpe Ratio': results['sharpe_ratio'],
                    'Win Rate %': results['win_rate'],
                    'Total Trades': results['total_trades']
                })
        except Exception as e:
            print(f"Error with {symbol}: {str(e)}")
    
    if results_summary:
        df = pd.DataFrame(results_summary)
        print("\nStrategy Performance Summary:")
        print(df.to_string(index=False, float_format='%.2f'))
        
        # Find best performing symbol
        best_symbol = df.loc[df['Return %'].idxmax(), 'Symbol']
        best_return = df.loc[df['Return %'].idxmax(), 'Return %']
        print(f"\nBest performing symbol: {best_symbol} with {best_return:.2f}% return")


def analyze_strategy_details():
    """Analyze detailed strategy performance for SPY."""
    
    print("\n" + "=" * 60)
    print("Detailed Analysis for SPY")
    print("=" * 60)
    
    backtester = Backtester(initial_capital=10000.0)
    
    try:
        results = backtester.backtest_close_to_open('SPY', period="6mo")
        
        if 'error' in results:
            print(f"Error: {results['error']}")
            return
        
        # Analyze portfolio value over time
        portfolio_history = results['portfolio_history']
        if portfolio_history:
            df = pd.DataFrame(portfolio_history)
            
            # Calculate some statistics
            max_value = df['portfolio_value'].max()
            min_value = df['portfolio_value'].min()
            max_drawdown = ((max_value - min_value) / max_value) * 100
            
            print(f"Portfolio Analysis:")
            print(f"Maximum Portfolio Value: ${max_value:,.2f}")
            print(f"Minimum Portfolio Value: ${min_value:,.2f}")
            print(f"Maximum Drawdown: {max_drawdown:.2f}%")
            
            # Show portfolio value progression (every 10th day)
            print(f"\nPortfolio Value Progression (sample):")
            sample_data = df[::10]  # Every 10th row
            for _, row in sample_data.iterrows():
                print(f"  {row['date'].strftime('%Y-%m-%d')}: ${row['portfolio_value']:,.2f}")
    
    except Exception as e:
        print(f"Error in detailed analysis: {str(e)}")


if __name__ == "__main__":
    try:
        # Test the strategy
        test_close_to_open_strategy()
        
        # Compare across symbols
        compare_symbols()
        
        # Detailed analysis
        analyze_strategy_details()
        
    except Exception as e:
        print(f"Error running backtest: {str(e)}")
        print("This is likely due to missing dependencies. Try installing: pip install yfinance pandas numpy")

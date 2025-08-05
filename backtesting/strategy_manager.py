"""
Strategy Manager

This module provides a unified interface for running and comparing different
investment strategies. It handles strategy execution, results analysis,
and performance visualization.
"""

import sys
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from strategies.buy_and_hold.buy_and_hold_strategy import BuyAndHoldStrategy

logger = logging.getLogger(__name__)

class StrategyManager:
    """
    Manager class for running and comparing investment strategies.
    """
    
    def __init__(self):
        """Initialize the strategy manager."""
        self.strategies = {
            'buy_and_hold': BuyAndHoldStrategy
        }
        self.results = []
        
    def register_strategy(self, name: str, strategy_class):
        """
        Register a new strategy class.
        
        Args:
            name: Strategy name identifier
            strategy_class: Strategy class (must inherit from BaseStrategy)
        """
        self.strategies[name] = strategy_class
        logger.info(f"Registered strategy: {name}")
    
    def run_strategy(self, strategy_name: str, ticker: str, 
                    start_date: Union[str, date], 
                    end_date: Optional[Union[str, date]] = None,
                    strategy_params: Optional[Dict] = None) -> Dict:
        """
        Run a specific strategy.
        
        Args:
            strategy_name: Name of strategy to run
            ticker: Stock ticker symbol
            start_date: Strategy start date
            end_date: Strategy end date (optional)
            strategy_params: Additional parameters for strategy initialization
            
        Returns:
            Strategy execution results
        """
        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found. "
                           f"Available strategies: {list(self.strategies.keys())}")
        
        # Initialize strategy with parameters
        strategy_params = strategy_params or {}
        strategy = self.strategies[strategy_name](**strategy_params)
        
        logger.info(f"Running {strategy_name} strategy for {ticker}")
        
        # Execute strategy
        result = strategy.execute_strategy(ticker, start_date, end_date)
        result['strategy_name'] = strategy_name
        
        # Store result
        self.results.append(result)
        
        return result
    
    def run_multiple_strategies(self, strategies: List[str], ticker: str,
                              start_date: Union[str, date],
                              end_date: Optional[Union[str, date]] = None,
                              strategy_params: Optional[Dict[str, Dict]] = None) -> List[Dict]:
        """
        Run multiple strategies for comparison.
        
        Args:
            strategies: List of strategy names to run
            ticker: Stock ticker symbol
            start_date: Strategy start date
            end_date: Strategy end date (optional)
            strategy_params: Parameters for each strategy {strategy_name: params}
            
        Returns:
            List of strategy results
        """
        strategy_params = strategy_params or {}
        results = []
        
        for strategy_name in strategies:
            try:
                params = strategy_params.get(strategy_name, {})
                result = self.run_strategy(strategy_name, ticker, start_date, end_date, params)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to run {strategy_name}: {e}")
                continue
        
        return results
    
    def compare_strategies(self, results: Optional[List[Dict]] = None) -> pd.DataFrame:
        """
        Compare performance of different strategies.
        
        Args:
            results: List of strategy results (uses stored results if None)
            
        Returns:
            DataFrame with strategy comparison
        """
        if results is None:
            results = self.results
        
        if not results:
            return pd.DataFrame()
        
        comparison_data = []
        
        for result in results:
            perf = result['performance']
            comparison_data.append({
                'Strategy': result.get('strategy_name', result.get('strategy', 'Unknown')),
                'Ticker': result['ticker'],
                'Start Date': result['start_date'],
                'End Date': result['end_date'],
                'Initial Capital': result['initial_capital'],
                'Final Value': perf.get('final_value', 0),
                'Total Return (%)': perf.get('total_return_pct', 0),
                'Annualized Return (%)': perf.get('annualized_return_pct', 0),
                'Volatility (%)': perf.get('volatility_pct', 0),
                'Sharpe Ratio': perf.get('sharpe_ratio', 0),
                'Max Drawdown (%)': perf.get('max_drawdown_pct', 0),
                'Total Trades': perf.get('total_trades', 0),
                'Days Invested': perf.get('days_invested', 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by total return descending
        comparison_df = comparison_df.sort_values('Total Return (%)', ascending=False)
        
        return comparison_df
    
    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy names."""
        return list(self.strategies.keys())
    
    def clear_results(self):
        """Clear stored results."""
        self.results = []
        logger.info("Cleared stored results")
    
    def save_results(self, filename: str):
        """
        Save results to CSV file.
        
        Args:
            filename: Output filename
        """
        if not self.results:
            logger.warning("No results to save")
            return
        
        comparison_df = self.compare_strategies()
        
        # Save comparison summary
        comparison_df.to_csv(f"data/{filename}_comparison.csv", index=False)
        
        # Save detailed results for each strategy
        for i, result in enumerate(self.results):
            strategy_name = result.get('strategy_name', f'strategy_{i}')
            
            # Save portfolio history
            if 'portfolio_history' in result and not result['portfolio_history'].empty:
                result['portfolio_history'].to_csv(
                    f"data/{filename}_{strategy_name}_portfolio.csv", index=False
                )
            
            # Save trades
            if 'trades' in result and not result['trades'].empty:
                result['trades'].to_csv(
                    f"data/{filename}_{strategy_name}_trades.csv", index=False
                )
        
        logger.info(f"Results saved with prefix: {filename}")


def interactive_strategy_runner():
    """Interactive command-line interface for running strategies."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Investment Strategy Manager")
    print("=" * 50)
    
    manager = StrategyManager()
    
    while True:
        print("\nAvailable Options:")
        print("1. Run Buy and Hold Strategy")
        print("2. Compare Multiple Strategies")
        print("3. View Previous Results")
        print("4. Save Results")
        print("5. Clear Results")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-5): ").strip()
        
        if choice == "0":
            print("👋 Goodbye!")
            break
        
        elif choice == "1":
            # Run Buy and Hold Strategy
            print("\n📊 Buy and Hold Strategy Setup")
            print("-" * 30)
            
            ticker = input("Enter ticker symbol (e.g., AAPL): ").strip().upper()
            start_date = input("Enter start date (YYYY-MM-DD): ").strip()
            end_date = input("Enter end date (YYYY-MM-DD, optional): ").strip()
            
            if not end_date:
                end_date = None
            
            try:
                capital = float(input("Enter initial capital (default 10000): ") or "10000")
                commission = float(input("Enter commission per trade (default 0): ") or "0")
                sell_at_end = input("Sell at end date? (y/n, default n): ").strip().lower() == 'y'
                
                params = {
                    'initial_capital': capital,
                    'commission': commission,
                    'sell_at_end': sell_at_end
                }
                
                print(f"\n🚀 Running Buy and Hold for {ticker}...")
                result = manager.run_strategy('buy_and_hold', ticker, start_date, end_date, params)
                
                # Display results
                print(f"\n📊 Results for {ticker}")
                print("-" * 30)
                perf = result['performance']
                print(f"Period: {result['start_date']} to {result['end_date']}")
                print(f"Initial Capital: ${result['initial_capital']:,.2f}")
                print(f"Final Value: ${perf['final_value']:,.2f}")
                print(f"Total Return: {perf['total_return_pct']:.2f}%")
                print(f"Annualized Return: {perf['annualized_return_pct']:.2f}%")
                print(f"Volatility: {perf['volatility_pct']:.2f}%")
                print(f"Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
                print(f"Max Drawdown: {perf['max_drawdown_pct']:.2f}%")
                print(f"Total Trades: {perf['total_trades']}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == "2":
            print("\n🔄 Multiple Strategy Comparison")
            print("-" * 35)
            print("Currently only Buy and Hold is available.")
            print("More strategies will be added in future updates.")
        
        elif choice == "3":
            # View previous results
            if not manager.results:
                print("\n📋 No previous results found.")
            else:
                comparison = manager.compare_strategies()
                print(f"\n📊 Strategy Comparison ({len(manager.results)} results)")
                print("-" * 50)
                print(comparison.to_string(index=False))
        
        elif choice == "4":
            # Save results
            if not manager.results:
                print("\n📋 No results to save.")
            else:
                filename = input("Enter filename prefix: ").strip()
                if filename:
                    manager.save_results(filename)
                    print(f"✅ Results saved with prefix: {filename}")
        
        elif choice == "5":
            # Clear results
            manager.clear_results()
            print("✅ Results cleared.")
        
        else:
            print("❌ Invalid choice. Please try again.")


if __name__ == "__main__":
    interactive_strategy_runner()

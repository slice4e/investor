"""
Buy and Hold Investment Strategy

A simple long-term investment strategy that buys a stock at a specified date
and holds it until the end of the backtesting period or a specified end date.

This strategy:
1. Buys maximum possible shares on the start date
2. Holds the position throughout the period
3. Sells all shares at the end date (if specified) or holds until end of data
"""

import sys
import os
from pathlib import Path
from datetime import datetime, date
from typing import Dict, Optional, Union
import pandas as pd
import numpy as np
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from backtesting.base_strategy import BaseStrategy
from data_downloader import StockDataDownloader

logger = logging.getLogger(__name__)

class BuyAndHoldStrategy(BaseStrategy):
    """
    Buy and Hold investment strategy.
    
    This strategy implements a simple buy-and-hold approach:
    - Buys maximum shares possible on the start date
    - Holds the position throughout the investment period
    - Optionally sells on a specified end date
    """
    
    def __init__(self, initial_capital: float = 10000.0, 
                 commission: float = 0.0, sell_at_end: bool = False):
        """
        Initialize Buy and Hold strategy.
        
        Args:
            initial_capital: Starting capital in dollars
            commission: Commission per trade (flat fee)
            sell_at_end: Whether to sell all positions at end date
        """
        super().__init__("Buy and Hold", initial_capital)
        self.commission = commission
        self.sell_at_end = sell_at_end
        self.downloader = StockDataDownloader()
        
    def ensure_data_available(self, ticker: str) -> pd.DataFrame:
        """
        Ensure stock data is available, download if necessary.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Historical stock data DataFrame
        """
        # Check if data exists in cache
        if ticker in self.data_cache:
            return self.data_cache[ticker]
        
        logger.info(f"Checking data availability for {ticker}")
        
        # Try to load existing data
        data_file = self.downloader.output_dir / "custom" / f"{ticker}_historical_data.csv"
        
        if data_file.exists():
            logger.info(f"Loading existing data for {ticker}")
            data = pd.read_csv(data_file)
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
        else:
            logger.info(f"Downloading historical data for {ticker}")
            # Use incremental download which will download full history if no data exists
            data = self.downloader.download_stock_data_incremental(ticker, "custom")
            if data is None:
                raise ValueError(f"Failed to download data for {ticker}")
            
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
        
        # Cache the data
        self.data_cache[ticker] = data
        logger.info(f"Data loaded for {ticker}: {len(data)} records from {data.index.min().date()} to {data.index.max().date()}")
        
        return data
    
    def generate_signals(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Generate trading signals for buy and hold strategy.
        
        Args:
            data: Historical price data
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with buy/hold signals
        """
        signals = data.copy()
        signals['signal'] = 'HOLD'
        
        # Buy signal on first available date
        signals.iloc[0, signals.columns.get_loc('signal')] = 'BUY'
        
        # Sell signal on last date if sell_at_end is True
        if self.sell_at_end and len(signals) > 1:
            signals.iloc[-1, signals.columns.get_loc('signal')] = 'SELL'
        
        return signals
    
    def execute_strategy(self, ticker: str, start_date: Union[str, date], 
                        end_date: Optional[Union[str, date]] = None) -> Dict:
        """
        Execute the buy and hold strategy.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Strategy start date
            end_date: Strategy end date (optional)
            
        Returns:
            Dictionary with strategy results and performance metrics
        """
        logger.info(f"Executing Buy and Hold strategy for {ticker}")
        logger.info(f"Start date: {start_date}, End date: {end_date}")
        logger.info(f"Initial capital: ${self.initial_capital:,.2f}")
        
        # Reset strategy state
        self.reset()
        
        # Ensure data is available
        data = self.ensure_data_available(ticker)
        
        # Convert dates to datetime
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).date()
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).date()
        
        # Filter data by date range
        mask = data.index.date >= start_date
        if end_date:
            mask = mask & (data.index.date <= end_date)
        
        strategy_data = data[mask].copy()
        
        if strategy_data.empty:
            raise ValueError(f"No data available for {ticker} in the specified date range")
        
        logger.info(f"Strategy period: {strategy_data.index.min().date()} to {strategy_data.index.max().date()}")
        logger.info(f"Total trading days: {len(strategy_data)}")
        
        # Generate signals
        signals = self.generate_signals(strategy_data, ticker)
        
        # Execute trades and track portfolio
        shares_owned = 0
        
        for current_date, row in signals.iterrows():
            price = row['Close']
            signal = row['signal']
            
            if signal == 'BUY' and shares_owned == 0:
                # Buy maximum shares possible
                affordable_shares = int((self.cash - self.commission) // price)
                if affordable_shares > 0:
                    success = self.buy_stock(ticker, affordable_shares, price, current_date)
                    if success:
                        shares_owned = affordable_shares
                        self.cash -= self.commission  # Apply commission
            
            elif signal == 'SELL' and shares_owned > 0:
                # Sell all shares
                success = self.sell_stock(ticker, shares_owned, price, current_date)
                if success:
                    shares_owned = 0
                    self.cash -= self.commission  # Apply commission
            
            # Calculate portfolio value
            portfolio_value = self.cash + (shares_owned * price)
            
            # Record portfolio history
            self.portfolio_history.append({
                'date': current_date,
                'ticker': ticker,
                'price': price,
                'shares': shares_owned,
                'cash': self.cash,
                'value': portfolio_value,
                'signal': signal
            })
        
        # Calculate performance metrics
        performance = self.calculate_performance_metrics()
        
        # Add strategy-specific metrics
        buy_price = None
        sell_price = None
        
        if self.trades:
            buy_trades = [t for t in self.trades if t['action'] == 'BUY']
            sell_trades = [t for t in self.trades if t['action'] == 'SELL']
            
            if buy_trades:
                buy_price = buy_trades[0]['price']
            if sell_trades:
                sell_price = sell_trades[-1]['price']
        
        # If still holding, use final price
        if shares_owned > 0:
            sell_price = strategy_data['Close'].iloc[-1]
        
        # Calculate buy and hold return
        if buy_price and sell_price:
            buy_hold_return = ((sell_price / buy_price) - 1) * 100
            performance['buy_hold_return_pct'] = round(buy_hold_return, 2)
        
        # Benchmark comparison (if data available)
        benchmark_return = self._calculate_benchmark_return(strategy_data)
        if benchmark_return is not None:
            performance['benchmark_return_pct'] = round(benchmark_return, 2)
            performance['excess_return_pct'] = round(
                performance['total_return_pct'] - benchmark_return, 2
            )
        
        result = {
            'strategy': self.name,
            'ticker': ticker,
            'start_date': strategy_data.index.min().date(),
            'end_date': strategy_data.index.max().date(),
            'initial_capital': self.initial_capital,
            'final_value': performance.get('final_value', 0),
            'shares_held': shares_owned,
            'buy_price': buy_price,
            'sell_price': sell_price,
            'performance': performance,
            'trades': self.get_trades_summary(),
            'portfolio_history': pd.DataFrame(self.portfolio_history)
        }
        
        # Log summary
        logger.info(f"Strategy completed successfully!")
        logger.info(f"Final portfolio value: ${result['final_value']:,.2f}")
        logger.info(f"Total return: {performance.get('total_return_pct', 0):.2f}%")
        if 'buy_hold_return_pct' in performance:
            logger.info(f"Buy & Hold return: {performance['buy_hold_return_pct']:.2f}%")
        
        return result
    
    def _calculate_benchmark_return(self, data: pd.DataFrame) -> Optional[float]:
        """Calculate benchmark return (stock price appreciation)."""
        if len(data) < 2:
            return None
        
        start_price = data['Close'].iloc[0]
        end_price = data['Close'].iloc[-1]
        return ((end_price / start_price) - 1) * 100


def run_buy_and_hold_example():
    """Example of running the Buy and Hold strategy."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Buy and Hold Strategy Example")
    print("=" * 50)
    
    # Initialize strategy
    strategy = BuyAndHoldStrategy(
        initial_capital=10000.0,
        commission=0.0,  # No commission for simplicity
        sell_at_end=False  # Hold until end
    )
    
    # Example parameters
    ticker = "AAPL"
    start_date = "2020-01-01"
    end_date = "2023-12-31"
    
    try:
        # Execute strategy
        result = strategy.execute_strategy(ticker, start_date, end_date)
        
        # Display results
        print(f"\n📊 Results for {result['ticker']} Buy and Hold Strategy")
        print("-" * 50)
        print(f"Period: {result['start_date']} to {result['end_date']}")
        print(f"Initial Capital: ${result['initial_capital']:,.2f}")
        print(f"Final Value: ${result['final_value']:,.2f}")
        print(f"Total Return: {result['performance']['total_return_pct']:.2f}%")
        print(f"Annualized Return: {result['performance']['annualized_return_pct']:.2f}%")
        print(f"Volatility: {result['performance']['volatility_pct']:.2f}%")
        print(f"Sharpe Ratio: {result['performance']['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {result['performance']['max_drawdown_pct']:.2f}%")
        
        if 'buy_hold_return_pct' in result['performance']:
            print(f"Stock Price Return: {result['performance']['buy_hold_return_pct']:.2f}%")
        
        print(f"\nTrades Executed: {result['performance']['total_trades']}")
        if not result['trades'].empty:
            print("\nTrade Details:")
            print(result['trades'].to_string(index=False))
        
    except Exception as e:
        print(f"❌ Error executing strategy: {e}")
        logger.error(f"Strategy execution failed: {e}", exc_info=True)


if __name__ == "__main__":
    run_buy_and_hold_example()

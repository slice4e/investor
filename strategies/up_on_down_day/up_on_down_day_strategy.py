"""
Up on Down Day Strategy

This contrarian investment strategy buys when the market has a "down day"
- defined as when a high percentage of stocks in an index are declining.

Strategy Logic:
1. Identify down days using configurable threshold (e.g., 80% of NASDAQ stocks down)
2. Buy target stock/ETF on down days
3. Hold for specified period or until next signal
4. Optional: Sell on "up days" (when high percentage of stocks are up)

This strategy is based on the contrarian principle that market-wide selling
often presents buying opportunities for strong stocks or index funds.
"""

import sys
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from backtesting.base_strategy import BaseStrategy
from data_downloader import StockDataDownloader
from strategies.up_on_down_day.market_analyzer import MarketAnalyzer

logger = logging.getLogger(__name__)

class UpOnDownDayStrategy(BaseStrategy):
    """
    Up on Down Day contrarian investment strategy.
    
    Buys stocks/ETFs when the market experiences down days (high percentage
    of stocks declining) and optionally sells on up days.
    """
    
    def __init__(self, initial_capital: float = 10000.0,
                 commission: float = 0.0,
                 down_day_threshold: float = 80.0,
                 up_day_threshold: float = 80.0,
                 index_for_signals: str = 'nasdaq',
                 hold_period_days: Optional[int] = None,
                 sell_on_up_days: bool = False,
                 max_positions: int = 1):
        """
        Initialize Up on Down Day strategy.
        
        Args:
            initial_capital: Starting capital in dollars
            commission: Commission per trade
            down_day_threshold: Percentage of stocks down to trigger buy
            up_day_threshold: Percentage of stocks up to trigger sell
            index_for_signals: Index to analyze ('nasdaq' or 'sp500')
            hold_period_days: Hold period in days (None = hold until sell signal)
            sell_on_up_days: Whether to sell on up days
            max_positions: Maximum number of positions to hold
        """
        super().__init__("Up on Down Day", initial_capital)
        self.commission = commission
        self.down_day_threshold = down_day_threshold
        self.up_day_threshold = up_day_threshold
        self.index_for_signals = index_for_signals
        self.hold_period_days = hold_period_days
        self.sell_on_up_days = sell_on_up_days
        self.max_positions = max_positions
        
        self.downloader = StockDataDownloader()
        self.market_analyzer = MarketAnalyzer()
        self.market_signals = None  # Will store daily market signals
        
    def prepare_market_signals(self, start_date: Union[str, date], 
                              end_date: Union[str, date]):
        """
        Prepare market-wide signals for the strategy period.
        
        Args:
            start_date: Strategy start date
            end_date: Strategy end date
        """
        logger.info(f"Preparing market signals for {self.index_for_signals.upper()}")
        logger.info(f"Down day threshold: {self.down_day_threshold}%")
        logger.info(f"Up day threshold: {self.up_day_threshold}%")
        
        # Get index data
        index_data = self.market_analyzer.ensure_index_data(self.index_for_signals)
        
        # Calculate daily market statistics
        daily_stats = self.market_analyzer.calculate_daily_down_percentage(
            index_data, start_date, end_date
        )
        
        # Generate buy/sell signals based on thresholds
        signals = daily_stats.copy()
        signals['signal'] = 'HOLD'
        
        # Buy signals on down days
        down_mask = signals['pct_down'] >= self.down_day_threshold
        signals.loc[down_mask, 'signal'] = 'BUY'
        
        # Sell signals on up days (if enabled)
        if self.sell_on_up_days:
            up_mask = signals['pct_up'] >= self.up_day_threshold
            signals.loc[up_mask, 'signal'] = 'SELL'
        
        self.market_signals = signals
        
        # Log signal summary
        buy_signals = len(signals[signals['signal'] == 'BUY'])
        sell_signals = len(signals[signals['signal'] == 'SELL'])
        logger.info(f"Generated {buy_signals} buy signals and {sell_signals} sell signals")
        
    def generate_signals(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Generate trading signals based on market down days.
        
        Args:
            data: Historical price data for target stock
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with buy/sell/hold signals
        """
        if self.market_signals is None:
            raise ValueError("Market signals not prepared. Call prepare_market_signals first.")
        
        # Align stock data with market signals
        signals = data.copy()
        signals['signal'] = 'HOLD'
        signals['market_pct_down'] = np.nan
        signals['market_pct_up'] = np.nan
        
        # Match dates between stock data and market signals
        for date in signals.index:
            date_key = date.date() if hasattr(date, 'date') else date
            
            if date_key in self.market_signals.index.date:
                market_row = self.market_signals[self.market_signals.index.date == date_key]
                if not market_row.empty:
                    signals.loc[date, 'signal'] = market_row['signal'].iloc[0]
                    signals.loc[date, 'market_pct_down'] = market_row['pct_down'].iloc[0]
                    signals.loc[date, 'market_pct_up'] = market_row['pct_up'].iloc[0]
        
        return signals
    
    def execute_strategy(self, ticker: str, start_date: Union[str, date], 
                        end_date: Optional[Union[str, date]] = None) -> Dict:
        """
        Execute the Up on Down Day strategy.
        
        Args:
            ticker: Target stock ticker to trade
            start_date: Strategy start date
            end_date: Strategy end date (optional)
            
        Returns:
            Dictionary with strategy results and performance metrics
        """
        logger.info(f"Executing Up on Down Day strategy for {ticker}")
        logger.info(f"Market signals from: {self.index_for_signals.upper()}")
        logger.info(f"Down day threshold: {self.down_day_threshold}%")
        
        # Reset strategy state
        self.reset()
        
        # Ensure target stock data is available
        logger.info(f"Ensuring data availability for target stock: {ticker}")
        target_data = self.downloader.download_stock_data_incremental(ticker, "custom")
        if target_data is None:
            raise ValueError(f"Failed to get data for {ticker}")
        
        target_data['Date'] = pd.to_datetime(target_data['Date'])
        target_data.set_index('Date', inplace=True)
        
        # Convert dates
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).date()
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).date()
        
        # Filter target data by date range
        mask = target_data.index.date >= start_date
        if end_date:
            mask = mask & (target_data.index.date <= end_date)
        
        strategy_data = target_data[mask].copy()
        
        if strategy_data.empty:
            raise ValueError(f"No data available for {ticker} in the specified date range")
        
        logger.info(f"Strategy period: {strategy_data.index.min().date()} to {strategy_data.index.max().date()}")
        
        # Prepare market signals for the same period
        self.prepare_market_signals(
            strategy_data.index.min().date(),
            strategy_data.index.max().date()
        )
        
        # Generate trading signals
        signals = self.generate_signals(strategy_data, ticker)
        
        # Execute trades and track portfolio
        shares_owned = 0
        last_buy_date = None
        
        for current_date, row in signals.iterrows():
            price = row['Close']
            signal = row['signal']
            market_pct_down = row.get('market_pct_down', np.nan)
            
            # Check hold period constraint
            hold_period_expired = False
            if (self.hold_period_days and last_buy_date and 
                (current_date.date() - last_buy_date).days >= self.hold_period_days):
                hold_period_expired = True
            
            # Execute buy signals
            if signal == 'BUY' and shares_owned == 0:
                # Buy maximum shares possible
                affordable_shares = int((self.cash - self.commission) // price)
                if affordable_shares > 0:
                    success = self.buy_stock(ticker, affordable_shares, price, current_date)
                    if success:
                        shares_owned = affordable_shares
                        last_buy_date = current_date.date()
                        self.cash -= self.commission
                        logger.info(f"BUY triggered: Market {market_pct_down:.1f}% down")
            
            # Execute sell signals
            elif ((signal == 'SELL' and self.sell_on_up_days) or hold_period_expired) and shares_owned > 0:
                success = self.sell_stock(ticker, shares_owned, price, current_date)
                if success:
                    shares_owned = 0
                    last_buy_date = None
                    self.cash -= self.commission
                    
                    if hold_period_expired:
                        logger.info(f"SELL triggered: Hold period expired")
                    else:
                        market_pct_up = row.get('market_pct_up', np.nan)
                        logger.info(f"SELL triggered: Market {market_pct_up:.1f}% up")
            
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
                'signal': signal,
                'market_pct_down': market_pct_down,
                'market_pct_up': row.get('market_pct_up', np.nan)
            })
        
        # Calculate performance metrics
        performance = self.calculate_performance_metrics()
        
        # Add strategy-specific metrics
        buy_trades = [t for t in self.trades if t['action'] == 'BUY']
        sell_trades = [t for t in self.trades if t['action'] == 'SELL']
        
        # Calculate average down day percentage on buy days
        avg_down_pct_on_buys = 0
        if buy_trades:
            buy_dates = [trade['date'].date() for trade in buy_trades]
            market_data_on_buys = self.market_signals[
                self.market_signals.index.date.isin(buy_dates)
            ]
            if not market_data_on_buys.empty:
                avg_down_pct_on_buys = market_data_on_buys['pct_down'].mean()
        
        performance['avg_market_down_on_buys'] = round(avg_down_pct_on_buys, 1)
        performance['down_day_threshold'] = self.down_day_threshold
        performance['market_index_used'] = self.index_for_signals.upper()
        
        # Count down days vs buy days
        total_down_days = len(self.market_signals[
            self.market_signals['pct_down'] >= self.down_day_threshold
        ])
        performance['total_down_days_in_period'] = total_down_days
        performance['buy_efficiency'] = (len(buy_trades) / total_down_days * 100) if total_down_days > 0 else 0
        
        result = {
            'strategy': self.name,
            'ticker': ticker,
            'start_date': strategy_data.index.min().date(),
            'end_date': strategy_data.index.max().date(),
            'initial_capital': self.initial_capital,
            'final_value': performance.get('final_value', 0),
            'shares_held': shares_owned,
            'performance': performance,
            'trades': self.get_trades_summary(),
            'portfolio_history': pd.DataFrame(self.portfolio_history),
            'market_signals': self.market_signals,
            'strategy_params': {
                'down_day_threshold': self.down_day_threshold,
                'up_day_threshold': self.up_day_threshold,
                'index_for_signals': self.index_for_signals,
                'hold_period_days': self.hold_period_days,
                'sell_on_up_days': self.sell_on_up_days
            }
        }
        
        # Log summary
        logger.info(f"Strategy completed!")
        logger.info(f"Final portfolio value: ${result['final_value']:,.2f}")
        logger.info(f"Total return: {performance.get('total_return_pct', 0):.2f}%")
        logger.info(f"Total down days in period: {total_down_days}")
        logger.info(f"Buy trades executed: {len(buy_trades)}")
        logger.info(f"Average market % down on buy days: {avg_down_pct_on_buys:.1f}%")
        
        return result


def run_up_on_down_day_example():
    """Example of running the Up on Down Day strategy."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("🚀 Up on Down Day Strategy Example")
    print("=" * 50)
    
    # Initialize strategy
    strategy = UpOnDownDayStrategy(
        initial_capital=10000.0,
        commission=0.0,
        down_day_threshold=80.0,  # Buy when 80%+ of NASDAQ stocks are down
        up_day_threshold=80.0,    # Sell when 80%+ of NASDAQ stocks are up
        index_for_signals='nasdaq',
        hold_period_days=None,    # Hold until sell signal
        sell_on_up_days=False,    # Don't sell on up days, just hold
        max_positions=1
    )
    
    # Example: Buy QQQ (NASDAQ ETF) on NASDAQ down days
    ticker = "QQQ"
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    
    try:
        # Execute strategy
        result = strategy.execute_strategy(ticker, start_date, end_date)
        
        # Display results
        print(f"\n📊 Results for {result['ticker']} Up on Down Day Strategy")
        print("-" * 60)
        print(f"Period: {result['start_date']} to {result['end_date']}")
        print(f"Market Signals: {result['strategy_params']['index_for_signals']}")
        print(f"Down Day Threshold: {result['strategy_params']['down_day_threshold']}%")
        print(f"Initial Capital: ${result['initial_capital']:,.2f}")
        print(f"Final Value: ${result['final_value']:,.2f}")
        print(f"Total Return: {result['performance']['total_return_pct']:.2f}%")
        print(f"Sharpe Ratio: {result['performance']['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {result['performance']['max_drawdown_pct']:.2f}%")
        
        print(f"\nStrategy-Specific Metrics:")
        print(f"Total Down Days in Period: {result['performance']['total_down_days_in_period']}")
        print(f"Buy Trades Executed: {result['performance']['total_trades']}")
        print(f"Average Market % Down on Buy Days: {result['performance']['avg_market_down_on_buys']:.1f}%")
        print(f"Buy Efficiency: {result['performance']['buy_efficiency']:.1f}%")
        
        if not result['trades'].empty:
            print(f"\nTrade Summary:")
            trades_df = result['trades']
            buy_trades = trades_df[trades_df['action'] == 'BUY']
            print(f"Number of Buy Trades: {len(buy_trades)}")
            if len(buy_trades) > 0:
                print(f"Average Buy Price: ${buy_trades['price'].mean():.2f}")
                print(f"First Buy: {buy_trades['date'].min().date()}")
                print(f"Last Buy: {buy_trades['date'].max().date()}")
        
    except Exception as e:
        print(f"❌ Error executing strategy: {e}")
        logger.error(f"Strategy execution failed: {e}", exc_info=True)


if __name__ == "__main__":
    run_up_on_down_day_example()

"""
Up on Down Day Strategy

This strategy identifies "down days" (when a high percentage of stocks are declining)
and purchases the top-performing "winner" stocks on those days, then sells them
the next trading day.

Strategy Logic:
1. Identify down days using a configurable threshold (e.g., 80% of stocks down)
2. On down days, find "winner" stocks that are up by a minimum threshold (e.g., 2%)
3. Purchase the top 3 winners at closing price on the down day
4. Sell all positions at opening price the next trading day

This is a contrarian strategy based on the premise that strong stocks on down days
may continue to outperform in the short term.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from backtesting.base_strategy import BaseStrategy
from strategies.up_on_down_day.market_analyzer import MarketAnalyzer

logger = logging.getLogger(__name__)


class UpOnDownDayStrategy(BaseStrategy):
    """
    Strategy that buys top winners on down days and sells the next day.
    """
    
    def __init__(self, initial_capital: float = 100000, 
                 down_day_threshold: float = 80.0,
                 winner_threshold: float = 2.0,
                 max_positions: int = 3,
                 position_size_pct: float = 0.33):
        """
        Initialize the Up on Down Day strategy.
        
        Args:
            initial_capital: Starting capital for the strategy
            down_day_threshold: Percentage of stocks down to qualify as "down day"
            winner_threshold: Minimum percentage gain to qualify as "winner"
            max_positions: Maximum number of positions to hold (top N winners)
            position_size_pct: Percentage of available capital to use per position
        """
        super().__init__("Up on Down Day", initial_capital)
        self.down_day_threshold = down_day_threshold
        self.winner_threshold = winner_threshold
        self.max_positions = max_positions
        self.position_size_pct = position_size_pct
        
        # Initialize market analyzer
        self.market_analyzer = MarketAnalyzer()
        self.market_data = None
        self.daily_stats = None
        
        # Track positions to sell next day
        self.positions_to_sell = []
        
        logger.info(f"Initialized UpOnDownDayStrategy:")
        logger.info(f"  Down day threshold: {down_day_threshold}%")
        logger.info(f"  Winner threshold: {winner_threshold}%")
        logger.info(f"  Max positions: {max_positions}")
        logger.info(f"  Position size: {position_size_pct*100}%")
    
    def generate_signals(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Generate trading signals for a given stock (required by BaseStrategy).
        
        This method is required by BaseStrategy but not used in this strategy
        since we use a different approach with market-wide analysis.
        
        Args:
            data: Historical price data for the stock
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with trading signals
        """
        # Create empty signals dataframe - not used in this strategy
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 'HOLD'
        return signals
    
    def execute_strategy(self, ticker: str = 'NASDAQ', start_date: str = '2024-01-01', 
                        end_date: str = '2024-12-31') -> Dict:
        """
        Execute the Up on Down Day strategy (BaseStrategy interface).
        
        Args:
            ticker: Ignored - we analyze the entire NASDAQ index
            start_date: Start date for backtesting (YYYY-MM-DD)
            end_date: End date for backtesting (YYYY-MM-DD)
            
        Returns:
            Dictionary with strategy results
        """
        return self.run_strategy(start_date, end_date)
    
    def run_strategy(self, start_date: str, end_date: str) -> Dict:
        """
        Execute the Up on Down Day strategy.
        
        Args:
            start_date: Start date for backtesting (YYYY-MM-DD)
            end_date: End date for backtesting (YYYY-MM-DD)
            
        Returns:
            Dictionary with strategy results
        """
        logger.info(f"Executing Up on Down Day strategy from {start_date} to {end_date}")
        
        # Prepare data
        self.prepare_data(start_date, end_date)
        
        # Get trading dates
        trading_dates = sorted(self.daily_stats.index.tolist())
        
        total_down_days = 0
        total_trades = 0
        
        for i, current_date in enumerate(trading_dates):
            # Process sells from previous day first
            self._process_sells(current_date)
            
            # Check if current day is a down day
            if self.is_down_day(current_date):
                total_down_days += 1
                logger.info(f"Down day detected: {current_date.strftime('%Y-%m-%d')}")
                
                # Find winners on this down day
                winners = self.find_winners_on_date(current_date)
                
                if winners:
                    # Select top winners (up to max_positions)
                    selected_winners = winners[:self.max_positions]
                    
                    logger.info(f"Found {len(winners)} winners, selecting top {len(selected_winners)}")
                    
                    # Execute buy orders at closing price
                    for winner in selected_winners:
                        success = self._execute_buy_order(winner, current_date)
                        if success:
                            total_trades += 1
                
                else:
                    logger.info(f"No winners found on down day {current_date.strftime('%Y-%m-%d')}")
        
        logger.info(f"Strategy execution completed:")
        logger.info(f"  Total down days: {total_down_days}")
        logger.info(f"  Total buy trades: {total_trades}")
        
        # Update current_capital to reflect final cash position (all positions should be closed)
        self.current_capital = self.cash
        logger.info(f"  Final portfolio value: ${self.current_capital:.2f}")
        
        return self.get_strategy_summary()
    
    def prepare_data(self, start_date: str, end_date: str):
        """
        Prepare market data for the strategy.
        
        Args:
            start_date: Start date for analysis (YYYY-MM-DD)
            end_date: End date for analysis (YYYY-MM-DD)
        """
        logger.info(f"Preparing market data from {start_date} to {end_date}")
        
        # Get NASDAQ market data
        self.market_data = self.market_analyzer.ensure_index_data('nasdaq')
        
        # Calculate daily market statistics
        self.daily_stats = self.market_analyzer.calculate_daily_down_percentage(
            self.market_data, start_date, end_date
        )
        
        logger.info(f"Prepared data for {len(self.daily_stats)} trading days")
        logger.info(f"Average % down: {self.daily_stats['pct_down'].mean():.1f}%")
    
    def is_down_day(self, date: pd.Timestamp) -> bool:
        """
        Check if a given date is a down day.
        
        Args:
            date: Date to check
            
        Returns:
            True if it's a down day, False otherwise
        """
        if date.date() not in self.daily_stats.index.date:
            return False
        
        daily_row = self.daily_stats[self.daily_stats.index.date == date.date()]
        if daily_row.empty:
            return False
        
        return daily_row['pct_down'].iloc[0] >= self.down_day_threshold
    
    def find_winners_on_date(self, date: pd.Timestamp) -> List[Dict]:
        """
        Find winner stocks on a specific date.
        
        Args:
            date: Date to analyze
            
        Returns:
            List of winner dictionaries sorted by performance
        """
        winners = []
        
        for ticker, stock_data in self.market_data.items():
            if date in stock_data.index:
                try:
                    day_data = stock_data.loc[date]
                    open_price = day_data['Open']
                    close_price = day_data['Close']
                    
                    if pd.notna(open_price) and pd.notna(close_price) and open_price > 0:
                        daily_return = (close_price - open_price) / open_price
                        
                        if daily_return >= (self.winner_threshold / 100.0):
                            winners.append({
                                'ticker': ticker,
                                'daily_return': daily_return,
                                'change_pct': daily_return * 100,
                                'open_price': open_price,
                                'close_price': close_price,
                                'volume': day_data.get('Volume', 0)
                            })
                
                except Exception as e:
                    logger.debug(f"Error processing {ticker} for {date}: {e}")
                    continue
        
        # Sort by performance (highest returns first)
        winners.sort(key=lambda x: x['daily_return'], reverse=True)
        
        return winners
    
    def execute_strategy(self, start_date: str, end_date: str):
        """
        Execute the Up on Down Day strategy.
        
        Args:
            start_date: Start date for backtesting (YYYY-MM-DD)
            end_date: End date for backtesting (YYYY-MM-DD)
        """
        logger.info(f"Executing Up on Down Day strategy from {start_date} to {end_date}")
        
        # Prepare data
        self.prepare_data(start_date, end_date)
        
        # Get trading dates
        trading_dates = sorted(self.daily_stats.index.tolist())
        
        total_down_days = 0
        total_trades = 0
        
        for i, current_date in enumerate(trading_dates):
            # Process sells from previous day first
            self._process_sells(current_date)
            
            # Check if current day is a down day
            if self.is_down_day(current_date):
                total_down_days += 1
                logger.info(f"Down day detected: {current_date.strftime('%Y-%m-%d')}")
                
                # Find winners on this down day
                winners = self.find_winners_on_date(current_date)
                
                if winners:
                    # Select top winners (up to max_positions)
                    selected_winners = winners[:self.max_positions]
                    
                    logger.info(f"Found {len(winners)} winners, selecting top {len(selected_winners)}")
                    
                    # Execute buy orders at closing price
                    for winner in selected_winners:
                        success = self._execute_buy_order(winner, current_date)
                        if success:
                            total_trades += 1
                
                else:
                    logger.info(f"No winners found on down day {current_date.strftime('%Y-%m-%d')}")
        
        logger.info(f"Strategy execution completed:")
        logger.info(f"  Total down days: {total_down_days}")
        logger.info(f"  Total buy trades: {total_trades}")
        logger.info(f"  Final portfolio value: ${self.portfolio_value:.2f}")
    
    def _execute_buy_order(self, winner: Dict, date: pd.Timestamp) -> bool:
        """
        Execute a buy order for a winner stock.
        
        Args:
            winner: Winner stock dictionary
            date: Trading date
            
        Returns:
            True if order was executed successfully
        """
        ticker = winner['ticker']
        price = winner['close_price']
        
        # Calculate position size
        available_cash = self.cash
        position_value = available_cash * self.position_size_pct
        shares = int(position_value / price)
        
        if shares > 0:
            success = self.buy_stock(ticker, shares, price, date)
            if success:
                logger.info(f"BUY: {shares} shares of {ticker} at ${price:.2f} "
                           f"(+{winner['change_pct']:.2f}% on down day)")
                
                # Schedule this position for sale next trading day
                self.positions_to_sell.append({
                    'ticker': ticker,
                    'shares': shares,
                    'buy_date': date,
                    'buy_price': price
                })
                
                return True
        
        return False
    
    def _process_sells(self, current_date: pd.Timestamp):
        """
        Process sell orders for positions bought on previous day.
        
        Args:
            current_date: Current trading date
        """
        if not self.positions_to_sell:
            return
        
        positions_sold = []
        
        for position in self.positions_to_sell[:]:  # Copy to avoid modification during iteration
            ticker = position['ticker']
            shares = position['shares']
            buy_price = position['buy_price']
            
            # Get opening price for current date
            if ticker in self.market_data and current_date in self.market_data[ticker].index:
                try:
                    sell_price = self.market_data[ticker].loc[current_date, 'Open']
                    
                    if pd.notna(sell_price) and sell_price > 0:
                        success = self.sell_stock(ticker, shares, sell_price, current_date)
                        
                        if success:
                            trade_return = (sell_price - buy_price) / buy_price * 100
                            logger.info(f"SELL: {shares} shares of {ticker} at ${sell_price:.2f} "
                                       f"(bought ${buy_price:.2f}, return: {trade_return:+.2f}%)")
                            
                            positions_sold.append(position)
                
                except Exception as e:
                    logger.warning(f"Error selling {ticker}: {e}")
                    positions_sold.append(position)  # Remove from queue anyway
        
        # Remove sold positions from queue
        for position in positions_sold:
            self.positions_to_sell.remove(position)
    
    def get_strategy_summary(self) -> Dict:
        """
        Get a summary of strategy performance and parameters.
        
        Returns:
            Dictionary with strategy summary
        """
        base_summary = self.calculate_performance_metrics()
        
        # Calculate additional metrics
        total_trades = len([t for t in self.trades if t['action'] == 'BUY'])
        winning_trades = len([t for t in self.trades if t['action'] == 'SELL' and self._calculate_trade_return(t) > 0])
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        # Calculate trade returns
        trade_returns = [self._calculate_trade_return(t) for t in self.trades if t['action'] == 'SELL']
        avg_trade_return = sum(trade_returns) / len(trade_returns) if trade_returns else 0
        best_trade = max(trade_returns) if trade_returns else 0
        worst_trade = min(trade_returns) if trade_returns else 0
        
        strategy_specific = {
            'strategy_name': 'Up on Down Day',
            'initial_capital': self.initial_capital,
            'final_portfolio_value': self.cash,  # Use cash since all positions are closed
            'total_return': ((self.cash - self.initial_capital) / self.initial_capital) * 100,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'win_rate': win_rate,
            'avg_trade_return': avg_trade_return,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'down_day_threshold': self.down_day_threshold,
            'winner_threshold': self.winner_threshold,
            'max_positions': self.max_positions,
            'position_size_pct': self.position_size_pct,
            'pending_sells': len(self.positions_to_sell)
        }
        
        return {**base_summary, **strategy_specific}
    
    def _calculate_trade_return(self, trade: Dict) -> float:
        """Calculate return percentage for a trade."""
        if trade['action'] != 'SELL':
            return 0
        
        # Find matching buy trade
        for buy_trade in reversed(self.trades):
            if (buy_trade['action'] == 'BUY' and 
                buy_trade['ticker'] == trade['ticker'] and
                buy_trade['date'] < trade['date']):
                return ((trade['price'] - buy_trade['price']) / buy_trade['price']) * 100
        
        return 0


def run_up_on_down_day_example():
    """Example of running the Up on Down Day strategy."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("📈 Up on Down Day Strategy Backtest")
    print("=" * 50)
    
    # Initialize strategy
    strategy = UpOnDownDayStrategy(
        initial_capital=100000,
        down_day_threshold=80.0,
        winner_threshold=2.0,
        max_positions=3,
        position_size_pct=0.30
    )
    
    # Run backtest for 2024
    try:
        summary = strategy.run_strategy('2024-01-01', '2024-12-31')
        
        print(f"\n📊 Strategy Results:")
        print("-" * 30)
        print(f"Initial Capital: ${summary['initial_capital']:,.2f}")
        print(f"Final Value: ${summary['final_portfolio_value']:,.2f}")
        print(f"Total Return: {summary['total_return']:+.2f}%")
        print(f"Total Trades: {summary['total_trades']}")
        print(f"Winning Trades: {summary['winning_trades']}")
        print(f"Win Rate: {summary['win_rate']:.1f}%")
        print(f"Average Trade: {summary['avg_trade_return']:+.2f}%")
        print(f"Best Trade: {summary['best_trade']:+.2f}%")
        print(f"Worst Trade: {summary['worst_trade']:+.2f}%")
        
        print(f"\n⚙️ Strategy Parameters:")
        print(f"Down Day Threshold: {summary['down_day_threshold']}%")
        print(f"Winner Threshold: {summary['winner_threshold']}%")
        print(f"Max Positions: {summary['max_positions']}")
        print(f"Position Size: {summary['position_size_pct']*100}%")
        
    except Exception as e:
        print(f"❌ Error running strategy: {e}")
        logger.error("Strategy execution failed", exc_info=True)


if __name__ == "__main__":
    run_up_on_down_day_example()
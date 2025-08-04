"""
Backtesting Framework Module

This module provides comprehensive backtesting functionality for trading strategies,
including historical data management, strategy execution, and performance analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from datetime import datetime, timedelta
import logging
from .data_fetcher import StockDataFetcher
from .portfolio import Portfolio


class BacktestEngine:
    """
    A comprehensive backtesting engine for trading strategies.
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        """
        Initialize the BacktestEngine.
        
        Args:
            initial_capital: Starting capital for backtesting
        """
        self.initial_capital = initial_capital
        self.data_fetcher = StockDataFetcher()
        self.logger = logging.getLogger(__name__)
        
    def get_intraday_data(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str,
        interval: str = "1m"
    ) -> pd.DataFrame:
        """
        Get intraday historical data for backtesting.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h)
        
        Returns:
            DataFrame with intraday data
        """
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(symbol)
            data = ticker.history(
                start=start_date,
                end=end_date,
                interval=interval,
                prepost=True  # Include pre/post market data
            )
            
            if data.empty:
                self.logger.warning(f"No intraday data found for {symbol}")
                return pd.DataFrame()
            
            # Add market session indicators
            data = self._add_market_session_indicators(data)
            
            self.logger.info(f"Retrieved {len(data)} intraday records for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching intraday data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def _add_market_session_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add indicators for market sessions (pre-market, regular, after-hours).
        
        Args:
            data: Intraday data DataFrame
        
        Returns:
            DataFrame with session indicators
        """
        data = data.copy()
        
        # Convert index to datetime if it's not already
        if not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)
        
        # Extract time component (assuming ET timezone)
        data['time'] = data.index.time
        data['date'] = data.index.date
        
        # Define market sessions (Eastern Time)
        pre_market_start = pd.Timestamp('04:00:00').time()
        market_open = pd.Timestamp('09:30:00').time()
        market_close = pd.Timestamp('16:00:00').time()
        after_hours_end = pd.Timestamp('20:00:00').time()
        
        # Create session indicators
        data['is_pre_market'] = (
            (data['time'] >= pre_market_start) & 
            (data['time'] < market_open)
        )
        data['is_regular_hours'] = (
            (data['time'] >= market_open) & 
            (data['time'] < market_close)
        )
        data['is_after_hours'] = (
            (data['time'] >= market_close) & 
            (data['time'] <= after_hours_end)
        )
        
        # Mark market close and next day open
        data['is_market_close'] = (data['time'] == market_close) | (
            (data['time'] < market_close) & 
            (data['time'].shift(-1) >= market_close)
        )
        data['is_market_open'] = (data['time'] == market_open) | (
            (data['time'] > market_open) & 
            (data['time'].shift(1) < market_open)
        )
        
        return data
    
    def backtest_intraday_strategy(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        strategy_func: Callable,
        initial_capital: Optional[float] = None
    ) -> Dict:
        """
        Backtest an intraday trading strategy.
        
        Args:
            symbol: Stock ticker symbol
            start_date: Start date for backtesting
            end_date: End date for backtesting
            strategy_func: Function that implements the trading strategy
            initial_capital: Starting capital (uses default if None)
        
        Returns:
            Dictionary with backtest results
        """
        if initial_capital is None:
            initial_capital = self.initial_capital
        
        # Get intraday data
        data = self.get_intraday_data(symbol, start_date, end_date, interval="1m")
        if data.empty:
            return {"error": f"No data available for {symbol}"}
        
        # Initialize portfolio for backtesting
        portfolio = Portfolio(initial_cash=initial_capital)
        
        # Track performance
        performance_log = []
        trades = []
        
        # Execute strategy
        for i, (timestamp, row) in enumerate(data.iterrows()):
            # Create market data context for strategy
            market_context = {
                'timestamp': timestamp,
                'current_data': row,
                'historical_data': data.iloc[:i+1] if i > 0 else data.iloc[:1],
                'symbol': symbol
            }
            
            # Execute strategy
            signal = strategy_func(market_context, portfolio)
            
            # Process trading signals
            if signal and 'action' in signal:
                if signal['action'] == 'BUY':
                    shares = signal.get('shares', 0)
                    if shares > 0:
                        success = portfolio.buy_stock(
                            symbol, 
                            shares, 
                            price=row['Close']
                        )
                        if success:
                            trades.append({
                                'timestamp': timestamp,
                                'action': 'BUY',
                                'symbol': symbol,
                                'shares': shares,
                                'price': row['Close'],
                                'session': self._get_session_type(row)
                            })
                
                elif signal['action'] == 'SELL':
                    shares = signal.get('shares', 0)
                    if shares > 0:
                        success = portfolio.sell_stock(
                            symbol, 
                            shares, 
                            price=row['Close']
                        )
                        if success:
                            trades.append({
                                'timestamp': timestamp,
                                'action': 'SELL',
                                'symbol': symbol,
                                'shares': shares,
                                'price': row['Close'],
                                'session': self._get_session_type(row)
                            })
            
            # Log portfolio performance periodically
            if i % 100 == 0:  # Every 100 data points
                portfolio_value = portfolio.get_portfolio_value()
                performance_log.append({
                    'timestamp': timestamp,
                    'portfolio_value': portfolio_value,
                    'cash': portfolio.cash,
                    'holdings_value': portfolio_value - portfolio.cash
                })
        
        # Calculate final performance metrics
        final_performance = self._calculate_performance_metrics(
            portfolio, trades, performance_log, initial_capital
        )
        
        return {
            'portfolio': portfolio,
            'trades': pd.DataFrame(trades) if trades else pd.DataFrame(),
            'performance_log': pd.DataFrame(performance_log),
            'metrics': final_performance,
            'data': data
        }
    
    def _get_session_type(self, row: pd.Series) -> str:
        """Get the market session type for a given row."""
        if row.get('is_pre_market', False):
            return 'pre_market'
        elif row.get('is_regular_hours', False):
            return 'regular_hours'
        elif row.get('is_after_hours', False):
            return 'after_hours'
        else:
            return 'unknown'
    
    def _calculate_performance_metrics(
        self, 
        portfolio: Portfolio, 
        trades: List[Dict], 
        performance_log: List[Dict],
        initial_capital: float
    ) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            portfolio: Final portfolio state
            trades: List of executed trades
            performance_log: Portfolio value over time
            initial_capital: Starting capital
        
        Returns:
            Dictionary with performance metrics
        """
        final_value = portfolio.get_portfolio_value()
        total_return = final_value - initial_capital
        total_return_pct = (total_return / initial_capital) * 100
        
        metrics = {
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'total_trades': len(trades),
            'cash_remaining': portfolio.cash
        }
        
        if trades:
            trades_df = pd.DataFrame(trades)
            
            # Calculate win rate
            buy_trades = trades_df[trades_df['action'] == 'BUY']
            sell_trades = trades_df[trades_df['action'] == 'SELL']
            
            if len(buy_trades) > 0 and len(sell_trades) > 0:
                # Match buy/sell pairs for P&L calculation
                trade_pairs = self._match_trade_pairs(trades_df)
                if trade_pairs:
                    winning_trades = sum(1 for pair in trade_pairs if pair['pnl'] > 0)
                    metrics['win_rate'] = (winning_trades / len(trade_pairs)) * 100
                    metrics['avg_trade_pnl'] = np.mean([pair['pnl'] for pair in trade_pairs])
                    metrics['total_trades_pairs'] = len(trade_pairs)
        
        # Calculate performance statistics from performance log
        if performance_log:
            perf_df = pd.DataFrame(performance_log)
            returns = perf_df['portfolio_value'].pct_change().dropna()
            
            if len(returns) > 0:
                metrics['volatility'] = returns.std() * np.sqrt(252 * 390)  # Annualized (390 min/day)
                metrics['sharpe_ratio'] = (
                    returns.mean() / returns.std() * np.sqrt(252 * 390) 
                    if returns.std() > 0 else 0
                )
                metrics['max_drawdown'] = self._calculate_max_drawdown(perf_df['portfolio_value'])
        
        return metrics
    
    def _match_trade_pairs(self, trades_df: pd.DataFrame) -> List[Dict]:
        """Match buy and sell trades to calculate P&L per trade pair."""
        trade_pairs = []
        position = 0
        avg_cost = 0
        
        for _, trade in trades_df.iterrows():
            if trade['action'] == 'BUY':
                if position == 0:
                    avg_cost = trade['price']
                else:
                    # Average down the cost
                    total_cost = (position * avg_cost) + (trade['shares'] * trade['price'])
                    total_shares = position + trade['shares']
                    avg_cost = total_cost / total_shares
                position += trade['shares']
            
            elif trade['action'] == 'SELL' and position > 0:
                shares_sold = min(trade['shares'], position)
                pnl = (trade['price'] - avg_cost) * shares_sold
                
                trade_pairs.append({
                    'buy_price': avg_cost,
                    'sell_price': trade['price'],
                    'shares': shares_sold,
                    'pnl': pnl,
                    'sell_timestamp': trade['timestamp']
                })
                
                position -= shares_sold
        
        return trade_pairs
    
    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown from portfolio values."""
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return drawdown.min() * 100  # Convert to percentage


class IntradayStrategy:
    """
    Implementation of the close-to-open intraday strategy.
    """
    
    @staticmethod
    def close_to_open_strategy(market_context: Dict, portfolio: Portfolio) -> Optional[Dict]:
        """
        Strategy: Buy at market close, sell at next market open.
        
        Args:
            market_context: Current market data and context
            portfolio: Current portfolio state
        
        Returns:
            Trading signal dictionary or None
        """
        current_data = market_context['current_data']
        symbol = market_context['symbol']
        
        # Check if we're at market close
        if current_data.get('is_market_close', False):
            # Calculate position size (use all available cash)
            current_price = current_data['Close']
            max_shares = int(portfolio.cash / current_price)
            
            if max_shares > 0:
                return {
                    'action': 'BUY',
                    'shares': max_shares,
                    'reason': 'Market close - entering position'
                }
        
        # Check if we're at market open and have position
        elif current_data.get('is_market_open', False):
            if symbol in portfolio.holdings:
                shares_to_sell = portfolio.holdings[symbol]['shares']
                if shares_to_sell > 0:
                    return {
                        'action': 'SELL',
                        'shares': shares_to_sell,
                        'reason': 'Market open - closing position'
                    }
        
        return None

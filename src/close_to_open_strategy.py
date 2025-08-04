"""
Close-to-Open Strategy Implementation

This module implements the specific close-to-open intraday trading strategy:
- Buy at market close
- Sell at next market open
- Execute this daily
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging

try:
    from .data_fetcher import StockDataFetcher
except ImportError:
    # For testing when modules aren't available
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_fetcher import StockDataFetcher


class CloseToOpenBacktester:
    """
    Specialized backtester for the close-to-open strategy.
    """
    
    def __init__(self, initial_capital: float = 10000.0):
        """
        Initialize the backtester.
        
        Args:
            initial_capital: Starting capital for backtesting
        """
        self.initial_capital = initial_capital
        self.data_fetcher = StockDataFetcher()
        self.logger = logging.getLogger(__name__)
    
    def get_daily_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """
        Get daily OHLCV data for the strategy.
        
        Args:
            symbol: Stock ticker symbol
            period: Data period
        
        Returns:
            DataFrame with daily data including next day's open
        """
        try:
            # Get daily data
            data = self.data_fetcher.get_stock_data(symbol, period, interval="1d")
            
            if data.empty:
                self.logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Add next day's open price (shifted)
            data = data.copy()
            data['Next_Open'] = data['Open'].shift(-1)
            
            # Remove the last row since it doesn't have a next open
            data = data[:-1].copy()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def calculate_performance_metrics(self, portfolio_values: List[Dict], trades: List[Dict]) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            portfolio_values: List of portfolio value records
            trades: List of trade records
        
        Returns:
            Dictionary with performance metrics
        """
        if not portfolio_values:
            return {}
        
        portfolio_df = pd.DataFrame(portfolio_values)
        portfolio_df['daily_return'] = portfolio_df['portfolio_value'].pct_change()
        
        final_value = portfolio_values[-1]['portfolio_value']
        total_return = final_value - self.initial_capital
        return_percentage = (total_return / self.initial_capital) * 100
        
        # Risk metrics
        daily_returns = portfolio_df['daily_return'].dropna()
        volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe_ratio = (daily_returns.mean() * 252) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        portfolio_df['cummax'] = portfolio_df['portfolio_value'].cummax()
        portfolio_df['drawdown'] = (portfolio_df['portfolio_value'] - portfolio_df['cummax']) / portfolio_df['cummax']
        max_drawdown = portfolio_df['drawdown'].min() * 100
        
        # Win rate calculation
        profitable_trades = 0
        total_trade_pairs = len(trades) // 2
        
        for i in range(0, len(trades) - 1, 2):
            if i + 1 < len(trades):
                buy_trade = trades[i]
                sell_trade = trades[i + 1]
                if sell_trade['value'] > buy_trade['value']:
                    profitable_trades += 1
        
        win_rate = (profitable_trades / total_trade_pairs * 100) if total_trade_pairs > 0 else 0
        
        return {
            'total_return': total_return,
            'return_percentage': return_percentage,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'total_trade_pairs': total_trade_pairs
        }
    
    def backtest_close_to_open(
        self, 
        symbol: str, 
        period: str = "1y",
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Backtest the close-to-open strategy.
        
        Strategy:
        1. Buy maximum shares possible at market close
        2. Sell all shares at next market open
        3. Repeat daily
        
        Args:
            symbol: Stock ticker symbol
            period: Data period if dates not specified
            start_date: Start date (YYYY-MM-DD) - optional
            end_date: End date (YYYY-MM-DD) - optional
        
        Returns:
            Dictionary with detailed backtest results
        """
        # Get historical data
        data = self.get_daily_data(symbol, period)
        
        if data.empty:
            return {"error": f"No data available for {symbol}"}
        
        # Filter by date range if specified
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        if data.empty:
            return {"error": "No data in specified date range"}
        
        # Initialize tracking variables
        capital = self.initial_capital
        shares = 0
        trades = []
        portfolio_values = []
        
        total_days = len(data)
        self.logger.info(f"Backtesting {symbol} close-to-open strategy for {total_days} days")
        
        for i, (date, row) in enumerate(data.iterrows()):
            close_price = row['Close']
            next_open_price = row['Next_Open']
            
            # Skip if we don't have next open price
            if pd.isna(next_open_price):
                continue
            
            # Step 1: Buy at close (use all available capital)
            if capital >= close_price:  # Can afford at least 1 share
                shares_to_buy = int(capital / close_price)
                if shares_to_buy > 0:
                    cost = shares_to_buy * close_price
                    capital -= cost
                    shares += shares_to_buy
                    
                    # Record buy transaction
                    trades.append({
                        'date': date,
                        'type': 'BUY',
                        'price': close_price,
                        'shares': shares_to_buy,
                        'value': cost
                    })
            
            # Step 2: Sell at next open
            if shares > 0:
                proceeds = shares * next_open_price
                capital += proceeds
                
                # Record sell transaction
                trades.append({
                    'date': date + timedelta(days=1),
                    'type': 'SELL',
                    'price': next_open_price,
                    'shares': shares,
                    'value': proceeds
                })
                
                shares = 0  # Reset shares after selling
            
            # Calculate current portfolio value
            current_portfolio_value = capital + (shares * close_price)
            portfolio_values.append({
                'date': date,
                'portfolio_value': current_portfolio_value,
                'capital': capital,
                'shares': shares,
                'stock_price': close_price
            })
        
        # Calculate performance metrics
        metrics = self.calculate_performance_metrics(portfolio_values, trades)
        
        # Buy and hold comparison
        if len(data) > 0:
            buy_hold_initial = data['Close'].iloc[0]
            buy_hold_final = data['Close'].iloc[-1]
            buy_hold_return = ((buy_hold_final - buy_hold_initial) / buy_hold_initial) * 100
            excess_return = metrics.get('return_percentage', 0) - buy_hold_return
        else:
            buy_hold_return = 0
            excess_return = 0
        
        # Compile results
        results = {
            'symbol': symbol,
            'strategy': 'Close-to-Open Daily',
            'period': f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}",
            'initial_capital': self.initial_capital,
            'final_value': portfolio_values[-1]['portfolio_value'] if portfolio_values else self.initial_capital,
            'buy_hold_return': buy_hold_return,
            'excess_return': excess_return,
            'trades': trades,
            'portfolio_history': portfolio_values,
            **metrics
        }
        
        self.logger.info(f"Backtest complete: {metrics.get('return_percentage', 0):.2f}% return, {metrics.get('win_rate', 0):.1f}% win rate")
        
        return results
    
    def compare_symbols(self, symbols: List[str], period: str = "1y") -> pd.DataFrame:
        """
        Compare the close-to-open strategy across multiple symbols.
        
        Args:
            symbols: List of stock symbols
            period: Data period
        
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for symbol in symbols:
            try:
                result = self.backtest_close_to_open(symbol, period)
                
                if 'error' not in result:
                    results.append({
                        'Symbol': symbol,
                        'Return %': result['return_percentage'],
                        'Buy & Hold %': result['buy_hold_return'],
                        'Excess Return %': result['excess_return'],
                        'Sharpe Ratio': result['sharpe_ratio'],
                        'Max Drawdown %': result['max_drawdown'],
                        'Win Rate %': result['win_rate'],
                        'Total Trades': result['total_trades']
                    })
                else:
                    self.logger.warning(f"Error with {symbol}: {result['error']}")
            
            except Exception as e:
                self.logger.error(f"Error testing {symbol}: {str(e)}")
        
        return pd.DataFrame(results) if results else pd.DataFrame()


# Alias for backward compatibility
Backtester = CloseToOpenBacktester
CloseToOpenStrategy = CloseToOpenBacktester  # For consistency with expected naming

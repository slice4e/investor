"""
Base Strategy Framework for Investment Backtesting

This module provides the abstract base class for all investment strategies.
All strategies should inherit from BaseStrategy and implement the required methods.
"""

from abc import ABC, abstractmethod
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    """
    Abstract base class for all investment strategies.
    
    This class defines the interface that all investment strategies must implement.
    It provides common functionality for data management, position tracking, and
    performance calculation.
    """
    
    def __init__(self, name: str, initial_capital: float = 10000.0):
        """
        Initialize the base strategy.
        
        Args:
            name: Name of the strategy
            initial_capital: Starting capital in dollars
        """
        self.name = name
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.positions = {}  # {ticker: shares}
        self.cash = initial_capital
        self.trades = []  # List of trade records
        self.portfolio_history = []  # Daily portfolio values
        self.data_cache = {}  # Cache for stock data
        
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """
        Generate trading signals for a given stock.
        
        Args:
            data: Historical price data for the stock
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with trading signals (buy/sell/hold)
        """
        pass
    
    @abstractmethod
    def execute_strategy(self, ticker: str, start_date: Union[str, date], 
                        end_date: Optional[Union[str, date]] = None) -> Dict:
        """
        Execute the strategy for a given ticker and date range.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Strategy start date
            end_date: Strategy end date (default: latest available)
            
        Returns:
            Dictionary with strategy results and performance metrics
        """
        pass
    
    def buy_stock(self, ticker: str, shares: int, price: float, date: datetime) -> bool:
        """
        Execute a buy order.
        
        Args:
            ticker: Stock ticker symbol
            shares: Number of shares to buy
            price: Price per share
            date: Transaction date
            
        Returns:
            True if successful, False otherwise
        """
        cost = shares * price
        
        if cost > self.cash:
            logger.warning(f"Insufficient funds to buy {shares} shares of {ticker}")
            return False
        
        # Execute trade
        self.cash -= cost
        self.positions[ticker] = self.positions.get(ticker, 0) + shares
        
        # Record trade
        trade = {
            'date': date,
            'ticker': ticker,
            'action': 'BUY',
            'shares': shares,
            'price': price,
            'value': cost,
            'cash_remaining': self.cash
        }
        self.trades.append(trade)
        
        logger.info(f"Bought {shares} shares of {ticker} at ${price:.2f} on {date.date()}")
        return True
    
    def sell_stock(self, ticker: str, shares: int, price: float, date: datetime) -> bool:
        """
        Execute a sell order.
        
        Args:
            ticker: Stock ticker symbol
            shares: Number of shares to sell
            price: Price per share
            date: Transaction date
            
        Returns:
            True if successful, False otherwise
        """
        if ticker not in self.positions or self.positions[ticker] < shares:
            logger.warning(f"Insufficient shares to sell {shares} shares of {ticker}")
            return False
        
        # Execute trade
        proceeds = shares * price
        self.cash += proceeds
        self.positions[ticker] -= shares
        
        # Remove ticker if no shares left
        if self.positions[ticker] == 0:
            del self.positions[ticker]
        
        # Record trade
        trade = {
            'date': date,
            'ticker': ticker,
            'action': 'SELL',
            'shares': shares,
            'price': price,
            'value': proceeds,
            'cash_remaining': self.cash
        }
        self.trades.append(trade)
        
        logger.info(f"Sold {shares} shares of {ticker} at ${price:.2f} on {date.date()}")
        return True
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Calculate current portfolio value.
        
        Args:
            current_prices: Dictionary of current stock prices
            
        Returns:
            Total portfolio value
        """
        stock_value = sum(
            shares * current_prices.get(ticker, 0)
            for ticker, shares in self.positions.items()
        )
        return self.cash + stock_value
    
    def calculate_performance_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        if not self.portfolio_history:
            return {}
        
        portfolio_df = pd.DataFrame(self.portfolio_history)
        portfolio_df['returns'] = portfolio_df['value'].pct_change()
        
        # Basic metrics
        total_return = (portfolio_df['value'].iloc[-1] / self.initial_capital - 1) * 100
        annualized_return = ((portfolio_df['value'].iloc[-1] / self.initial_capital) ** 
                           (252 / len(portfolio_df)) - 1) * 100
        
        # Risk metrics
        volatility = portfolio_df['returns'].std() * np.sqrt(252) * 100
        sharpe_ratio = (annualized_return / 100) / (volatility / 100) if volatility > 0 else 0
        
        # Drawdown analysis
        rolling_max = portfolio_df['value'].expanding().max()
        drawdown = (portfolio_df['value'] - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()
        
        # Win rate (if we have trades)
        win_rate = 0
        if self.trades:
            profitable_trades = sum(1 for trade in self.trades 
                                  if trade['action'] == 'SELL' and 
                                  self._calculate_trade_profit(trade) > 0)
            sell_trades = sum(1 for trade in self.trades if trade['action'] == 'SELL')
            win_rate = (profitable_trades / sell_trades * 100) if sell_trades > 0 else 0
        
        return {
            'total_return_pct': round(total_return, 2),
            'annualized_return_pct': round(annualized_return, 2),
            'volatility_pct': round(volatility, 2),
            'sharpe_ratio': round(sharpe_ratio, 2),
            'max_drawdown_pct': round(max_drawdown, 2),
            'win_rate_pct': round(win_rate, 2),
            'total_trades': len(self.trades),
            'final_value': round(portfolio_df['value'].iloc[-1], 2),
            'days_invested': len(portfolio_df)
        }
    
    def _calculate_trade_profit(self, sell_trade: Dict) -> float:
        """Calculate profit from a sell trade (simplified)."""
        # This is a simplified calculation - in practice, you'd match with buy trades
        return 0
    
    def get_trades_summary(self) -> pd.DataFrame:
        """
        Get summary of all trades.
        
        Returns:
            DataFrame with trade details
        """
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades)
    
    def reset(self):
        """Reset strategy to initial state."""
        self.current_capital = self.initial_capital
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.portfolio_history = []

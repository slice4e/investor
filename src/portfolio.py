"""
Portfolio Management Module

This module provides functionality for managing investment portfolios,
tracking holdings, calculating performance metrics, and portfolio optimization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from .data_fetcher import StockDataFetcher


class Portfolio:
    """
    A class to manage and analyze investment portfolios.
    """
    
    def __init__(self, initial_cash: float = 10000.0):
        """
        Initialize the Portfolio.
        
        Args:
            initial_cash: Initial cash amount for the portfolio
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.holdings: Dict[str, Dict] = {}  # {symbol: {shares: int, avg_cost: float}}
        self.transactions: List[Dict] = []
        self.data_fetcher = StockDataFetcher()
        self.logger = logging.getLogger(__name__)
    
    def buy_stock(self, symbol: str, shares: int, price: Optional[float] = None) -> bool:
        """
        Buy shares of a stock.
        
        Args:
            symbol: Stock ticker symbol
            shares: Number of shares to buy
            price: Price per share (if None, fetches current price)
        
        Returns:
            True if transaction successful, False otherwise
        """
        if price is None:
            price = self.data_fetcher.get_real_time_price(symbol)
            if price is None:
                self.logger.error(f"Could not fetch price for {symbol}")
                return False
        
        total_cost = shares * price
        
        if total_cost > self.cash:
            self.logger.warning(f"Insufficient funds. Need ${total_cost:.2f}, have ${self.cash:.2f}")
            return False
        
        # Update cash
        self.cash -= total_cost
        
        # Update holdings
        if symbol in self.holdings:
            current_shares = self.holdings[symbol]['shares']
            current_value = current_shares * self.holdings[symbol]['avg_cost']
            new_avg_cost = (current_value + total_cost) / (current_shares + shares)
            
            self.holdings[symbol]['shares'] += shares
            self.holdings[symbol]['avg_cost'] = new_avg_cost
        else:
            self.holdings[symbol] = {'shares': shares, 'avg_cost': price}
        
        # Record transaction
        transaction = {
            'timestamp': datetime.now(),
            'type': 'BUY',
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'total': total_cost
        }
        self.transactions.append(transaction)
        
        self.logger.info(f"Bought {shares} shares of {symbol} at ${price:.2f}")
        return True
    
    def sell_stock(self, symbol: str, shares: int, price: Optional[float] = None) -> bool:
        """
        Sell shares of a stock.
        
        Args:
            symbol: Stock ticker symbol
            shares: Number of shares to sell
            price: Price per share (if None, fetches current price)
        
        Returns:
            True if transaction successful, False otherwise
        """
        if symbol not in self.holdings:
            self.logger.warning(f"No holdings found for {symbol}")
            return False
        
        if shares > self.holdings[symbol]['shares']:
            self.logger.warning(f"Insufficient shares. Have {self.holdings[symbol]['shares']}, trying to sell {shares}")
            return False
        
        if price is None:
            price = self.data_fetcher.get_real_time_price(symbol)
            if price is None:
                self.logger.error(f"Could not fetch price for {symbol}")
                return False
        
        total_value = shares * price
        
        # Update cash
        self.cash += total_value
        
        # Update holdings
        self.holdings[symbol]['shares'] -= shares
        if self.holdings[symbol]['shares'] == 0:
            del self.holdings[symbol]
        
        # Record transaction
        transaction = {
            'timestamp': datetime.now(),
            'type': 'SELL',
            'symbol': symbol,
            'shares': shares,
            'price': price,
            'total': total_value
        }
        self.transactions.append(transaction)
        
        self.logger.info(f"Sold {shares} shares of {symbol} at ${price:.2f}")
        return True
    
    def get_portfolio_value(self) -> float:
        """
        Calculate total portfolio value.
        
        Returns:
            Total portfolio value (cash + holdings)
        """
        holdings_value = 0
        
        for symbol, holding in self.holdings.items():
            current_price = self.data_fetcher.get_real_time_price(symbol)
            if current_price:
                holdings_value += holding['shares'] * current_price
        
        return self.cash + holdings_value
    
    def get_portfolio_performance(self) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        current_value = self.get_portfolio_value()
        total_return = current_value - self.initial_cash
        return_percentage = (total_return / self.initial_cash) * 100
        
        return {
            'initial_value': self.initial_cash,
            'current_value': current_value,
            'total_return': total_return,
            'return_percentage': return_percentage,
            'cash': self.cash
        }
    
    def get_holdings_summary(self) -> pd.DataFrame:
        """
        Get summary of current holdings.
        
        Returns:
            DataFrame with holdings information
        """
        if not self.holdings:
            return pd.DataFrame()
        
        holdings_data = []
        
        for symbol, holding in self.holdings.items():
            current_price = self.data_fetcher.get_real_time_price(symbol)
            if current_price:
                current_value = holding['shares'] * current_price
                cost_basis = holding['shares'] * holding['avg_cost']
                unrealized_pnl = current_value - cost_basis
                unrealized_pnl_pct = (unrealized_pnl / cost_basis) * 100
                
                holdings_data.append({
                    'Symbol': symbol,
                    'Shares': holding['shares'],
                    'Avg Cost': holding['avg_cost'],
                    'Current Price': current_price,
                    'Current Value': current_value,
                    'Cost Basis': cost_basis,
                    'Unrealized P&L': unrealized_pnl,
                    'Unrealized P&L %': unrealized_pnl_pct
                })
        
        return pd.DataFrame(holdings_data)
    
    def get_transaction_history(self) -> pd.DataFrame:
        """
        Get transaction history as DataFrame.
        
        Returns:
            DataFrame with transaction history
        """
        if not self.transactions:
            return pd.DataFrame()
        
        return pd.DataFrame(self.transactions)

"""
Stock Data Fetcher Module

This module provides functionality to fetch stock data from various sources
including Yahoo Finance, Alpha Vantage, and other financial APIs.
"""

import yfinance as yf
import pandas as pd
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
import logging


class StockDataFetcher:
    """
    A class to fetch stock data from various financial data sources.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the StockDataFetcher.
        
        Args:
            api_key: Optional API key for premium data sources
        """
        self.api_key = api_key
        self.logger = logging.getLogger(__name__)
    
    def get_stock_data(
        self, 
        symbol: str, 
        period: str = "1y",
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch stock data for a given symbol.
        
        Args:
            symbol: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        
        Returns:
            DataFrame with stock data (Open, High, Low, Close, Volume)
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                self.logger.warning(f"No data found for symbol {symbol}")
                return pd.DataFrame()
            
            self.logger.info(f"Successfully fetched data for {symbol}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_multiple_stocks(
        self, 
        symbols: List[str], 
        period: str = "1y"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch stock data for multiple symbols.
        
        Args:
            symbols: List of stock ticker symbols
            period: Data period
        
        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        results = {}
        for symbol in symbols:
            data = self.get_stock_data(symbol, period)
            if not data.empty:
                results[symbol] = data
        
        return results
    
    def get_stock_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get comprehensive stock information.
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            Dictionary with stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return info
        except Exception as e:
            self.logger.error(f"Error fetching info for {symbol}: {str(e)}")
            return {}
    
    def get_real_time_price(self, symbol: str) -> Optional[float]:
        """
        Get real-time stock price.
        
        Args:
            symbol: Stock ticker symbol
        
        Returns:
            Current stock price or None if error
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="1d", interval="1m")
            if not data.empty:
                return float(data['Close'].iloc[-1])
            return None
        except Exception as e:
            self.logger.error(f"Error fetching real-time price for {symbol}: {str(e)}")
            return None

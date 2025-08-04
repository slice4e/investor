"""
Stock Analysis Module

This module provides comprehensive stock analysis functionality including
technical indicators, fundamental analysis, and visualization tools.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import logging
from .data_fetcher import StockDataFetcher


class StockAnalyzer:
    """
    A class for analyzing stocks using technical and fundamental analysis.
    """
    
    def __init__(self):
        """Initialize the StockAnalyzer."""
        self.data_fetcher = StockDataFetcher()
        self.logger = logging.getLogger(__name__)
    
    def calculate_moving_averages(
        self, 
        data: pd.DataFrame, 
        windows: List[int] = [20, 50, 200]
    ) -> pd.DataFrame:
        """
        Calculate moving averages for stock data.
        
        Args:
            data: Stock price DataFrame
            windows: List of window sizes for moving averages
        
        Returns:
            DataFrame with moving averages added
        """
        result = data.copy()
        
        for window in windows:
            col_name = f'MA_{window}'
            result[col_name] = result['Close'].rolling(window=window).mean()
        
        return result
    
    def calculate_rsi(self, data: pd.DataFrame, window: int = 14) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            data: Stock price DataFrame
            window: RSI calculation window
        
        Returns:
            RSI values as Series
        """
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_bollinger_bands(
        self, 
        data: pd.DataFrame, 
        window: int = 20, 
        num_std: float = 2
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Args:
            data: Stock price DataFrame
            window: Moving average window
            num_std: Number of standard deviations for bands
        
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        middle_band = data['Close'].rolling(window=window).mean()
        std = data['Close'].rolling(window=window).std()
        
        upper_band = middle_band + (std * num_std)
        lower_band = middle_band - (std * num_std)
        
        return upper_band, middle_band, lower_band
    
    def calculate_macd(
        self, 
        data: pd.DataFrame, 
        fast: int = 12, 
        slow: int = 26, 
        signal: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            data: Stock price DataFrame
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line EMA period
        
        Returns:
            Tuple of (macd_line, signal_line, histogram)
        """
        ema_fast = data['Close'].ewm(span=fast).mean()
        ema_slow = data['Close'].ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def calculate_volatility(self, data: pd.DataFrame, window: int = 30) -> pd.Series:
        """
        Calculate rolling volatility.
        
        Args:
            data: Stock price DataFrame
            window: Rolling window for volatility calculation
        
        Returns:
            Volatility values as Series
        """
        returns = data['Close'].pct_change()
        volatility = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        
        return volatility
    
    def analyze_stock(self, symbol: str, period: str = "1y") -> Dict:
        """
        Perform comprehensive stock analysis.
        
        Args:
            symbol: Stock ticker symbol
            period: Data period for analysis
        
        Returns:
            Dictionary with analysis results
        """
        # Fetch data
        data = self.data_fetcher.get_stock_data(symbol, period)
        if data.empty:
            return {}
        
        # Calculate technical indicators
        data_with_ma = self.calculate_moving_averages(data)
        rsi = self.calculate_rsi(data)
        upper_bb, middle_bb, lower_bb = self.calculate_bollinger_bands(data)
        macd_line, signal_line, histogram = self.calculate_macd(data)
        volatility = self.calculate_volatility(data)
        
        # Calculate basic metrics
        current_price = data['Close'].iloc[-1]
        price_change = data['Close'].iloc[-1] - data['Close'].iloc[-2]
        price_change_pct = (price_change / data['Close'].iloc[-2]) * 100
        
        # Volume analysis
        avg_volume = data['Volume'].mean()
        current_volume = data['Volume'].iloc[-1]
        volume_ratio = current_volume / avg_volume
        
        # Price ranges
        week_52_high = data['High'].max()
        week_52_low = data['Low'].min()
        
        analysis_result = {
            'symbol': symbol,
            'current_price': current_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'volume_ratio': volume_ratio,
            '52_week_high': week_52_high,
            '52_week_low': week_52_low,
            'current_rsi': rsi.iloc[-1] if not rsi.empty else None,
            'current_volatility': volatility.iloc[-1] if not volatility.empty else None,
            'ma_20': data_with_ma['MA_20'].iloc[-1] if 'MA_20' in data_with_ma else None,
            'ma_50': data_with_ma['MA_50'].iloc[-1] if 'MA_50' in data_with_ma else None,
            'ma_200': data_with_ma['MA_200'].iloc[-1] if 'MA_200' in data_with_ma else None,
            'data': data_with_ma
        }
        
        return analysis_result
    
    def plot_stock_analysis(self, symbol: str, period: str = "1y") -> None:
        """
        Create comprehensive stock analysis plots.
        
        Args:
            symbol: Stock ticker symbol
            period: Data period for analysis
        """
        analysis = self.analyze_stock(symbol, period)
        if not analysis:
            self.logger.error(f"No data available for {symbol}")
            return
        
        data = analysis['data']
        
        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        fig.suptitle(f'{symbol} Stock Analysis', fontsize=16)
        
        # Price and moving averages
        axes[0].plot(data.index, data['Close'], label='Close Price', linewidth=2)
        if 'MA_20' in data:
            axes[0].plot(data.index, data['MA_20'], label='MA 20', alpha=0.7)
        if 'MA_50' in data:
            axes[0].plot(data.index, data['MA_50'], label='MA 50', alpha=0.7)
        if 'MA_200' in data:
            axes[0].plot(data.index, data['MA_200'], label='MA 200', alpha=0.7)
        
        axes[0].set_title('Price and Moving Averages')
        axes[0].set_ylabel('Price ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Volume
        axes[1].bar(data.index, data['Volume'], alpha=0.7, color='orange')
        axes[1].set_title('Volume')
        axes[1].set_ylabel('Volume')
        axes[1].grid(True, alpha=0.3)
        
        # RSI
        rsi = self.calculate_rsi(data)
        axes[2].plot(data.index, rsi, label='RSI', color='purple')
        axes[2].axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
        axes[2].axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
        axes[2].set_title('Relative Strength Index (RSI)')
        axes[2].set_ylabel('RSI')
        axes[2].set_xlabel('Date')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def compare_stocks(self, symbols: List[str], period: str = "1y") -> pd.DataFrame:
        """
        Compare multiple stocks performance.
        
        Args:
            symbols: List of stock ticker symbols
            period: Data period for comparison
        
        Returns:
            DataFrame with comparison metrics
        """
        comparison_data = []
        
        for symbol in symbols:
            analysis = self.analyze_stock(symbol, period)
            if analysis:
                comparison_data.append({
                    'Symbol': symbol,
                    'Current Price': analysis['current_price'],
                    'Price Change %': analysis['price_change_pct'],
                    '52W High': analysis['52_week_high'],
                    '52W Low': analysis['52_week_low'],
                    'RSI': analysis['current_rsi'],
                    'Volatility': analysis['current_volatility'],
                    'Volume Ratio': analysis['volume_ratio']
                })
        
        return pd.DataFrame(comparison_data)

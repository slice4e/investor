"""
Test module for StockDataFetcher
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from src.data_fetcher import StockDataFetcher


class TestStockDataFetcher:
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fetcher = StockDataFetcher()
    
    def test_init(self):
        """Test StockDataFetcher initialization."""
        assert self.fetcher.api_key is None
        
        fetcher_with_key = StockDataFetcher(api_key="test_key")
        assert fetcher_with_key.api_key == "test_key"
    
    @patch('src.data_fetcher.yf.Ticker')
    def test_get_stock_data_success(self, mock_ticker):
        """Test successful stock data retrieval."""
        # Mock data
        mock_data = pd.DataFrame({
            'Open': [100, 101, 102],
            'High': [105, 106, 107],
            'Low': [99, 100, 101],
            'Close': [104, 105, 106],
            'Volume': [1000, 1100, 1200]
        })
        
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = mock_data
        mock_ticker.return_value = mock_ticker_instance
        
        result = self.fetcher.get_stock_data("AAPL")
        
        assert isinstance(result, pd.DataFrame)
        assert not result.empty
        mock_ticker.assert_called_once_with("AAPL")
        mock_ticker_instance.history.assert_called_once_with(period="1y", interval="1d")
    
    @patch('src.data_fetcher.yf.Ticker')
    def test_get_stock_data_empty(self, mock_ticker):
        """Test handling of empty stock data."""
        mock_ticker_instance = Mock()
        mock_ticker_instance.history.return_value = pd.DataFrame()
        mock_ticker.return_value = mock_ticker_instance
        
        result = self.fetcher.get_stock_data("INVALID")
        
        assert isinstance(result, pd.DataFrame)
        assert result.empty
    
    def test_get_multiple_stocks(self):
        """Test fetching data for multiple stocks."""
        with patch.object(self.fetcher, 'get_stock_data') as mock_get_data:
            mock_data = pd.DataFrame({'Close': [100, 101, 102]})
            mock_get_data.return_value = mock_data
            
            symbols = ["AAPL", "GOOGL", "MSFT"]
            result = self.fetcher.get_multiple_stocks(symbols)
            
            assert len(result) == 3
            assert all(symbol in result for symbol in symbols)
            assert mock_get_data.call_count == 3

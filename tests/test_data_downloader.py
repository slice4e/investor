"""
Test suite for the Stock Data Downloader

Tests core functionality of downloading and exporting stock data.
"""

import pytest
import pandas as pd
import os
import tempfile
from data_downloader import StockDataDownloader

class TestStockDataDownloader:
    
    def setup_method(self):
        """Set up test fixtures with temporary directory."""
        self.temp_dir = tempfile.mkdtemp()
        self.downloader = StockDataDownloader(output_dir=self.temp_dir)
    
    def test_download_single_stock(self):
        """Test downloading a single stock."""
        data = self.downloader.download_stock_data('AAPL', period='5d')
        
        assert data is not None
        assert not data.empty
        assert 'Ticker' in data.columns
        assert data['Ticker'].iloc[0] == 'AAPL'
        assert 'Date' in data.columns
        assert 'Close' in data.columns
    
    def test_download_custom_tickers(self):
        """Test downloading custom list of tickers."""
        tickers = ['AAPL', 'MSFT']
        data = self.downloader.download_custom_tickers(tickers)
        
        assert len(data) == 2
        assert 'AAPL' in data
        assert 'MSFT' in data
        
        for ticker, df in data.items():
            assert not df.empty
            assert df['Ticker'].iloc[0] == ticker
    
    def test_save_to_csv(self):
        """Test saving data to CSV files."""
        # Download test data
        tickers = ['AAPL']
        data = self.downloader.download_custom_tickers(tickers)
        
        # Save to CSV
        self.downloader.save_to_csv(data, "test_csv")
        
        # Check file was created
        csv_file = os.path.join(self.temp_dir, "test_csv", "AAPL_historical_data.csv")
        assert os.path.exists(csv_file)
        
        # Verify content
        df = pd.read_csv(csv_file)
        assert not df.empty
        assert 'Ticker' in df.columns
    
    def test_create_combined_dataset(self):
        """Test creating combined dataset from multiple stocks."""
        tickers = ['AAPL', 'MSFT']
        data = self.downloader.download_custom_tickers(tickers)
        
        combined_df = self.downloader.create_combined_dataset(data)
        
        assert not combined_df.empty
        assert 'Ticker' in combined_df.columns
        assert combined_df['Ticker'].nunique() == 2
        assert 'AAPL' in combined_df['Ticker'].values
        assert 'MSFT' in combined_df['Ticker'].values
    
    def test_get_sp500_tickers(self):
        """Test fetching S&P 500 ticker list."""
        tickers = self.downloader.get_sp500_tickers()
        
        assert isinstance(tickers, list)
        assert len(tickers) > 400  # Should have at least 400 tickers
        assert 'AAPL' in tickers  # Apple should be in S&P 500
    
    def test_invalid_ticker(self):
        """Test handling of invalid ticker."""
        data = self.downloader.download_stock_data('INVALID_TICKER_XYZ')
        
        # Should return None for invalid ticker
        assert data is None
    
    def teardown_method(self):
        """Clean up temporary files."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

if __name__ == "__main__":
    pytest.main([__file__])

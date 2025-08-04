"""
Test module for Portfolio
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch
from src.portfolio import Portfolio


class TestPortfolio:
    
    def setup_method(self):
        """Set up test fixtures."""
        self.portfolio = Portfolio(initial_cash=10000.0)
    
    def test_init(self):
        """Test Portfolio initialization."""
        assert self.portfolio.initial_cash == 10000.0
        assert self.portfolio.cash == 10000.0
        assert self.portfolio.holdings == {}
        assert self.portfolio.transactions == []
    
    @patch('src.portfolio.StockDataFetcher')
    def test_buy_stock_success(self, mock_fetcher_class):
        """Test successful stock purchase."""
        # Mock the data fetcher
        mock_fetcher = Mock()
        mock_fetcher.get_real_time_price.return_value = 100.0
        mock_fetcher_class.return_value = mock_fetcher
        
        # Create fresh portfolio instance
        portfolio = Portfolio(initial_cash=10000.0)
        
        result = portfolio.buy_stock("AAPL", 10)
        
        assert result is True
        assert portfolio.cash == 9000.0  # 10000 - (10 * 100)
        assert "AAPL" in portfolio.holdings
        assert portfolio.holdings["AAPL"]["shares"] == 10
        assert portfolio.holdings["AAPL"]["avg_cost"] == 100.0
        assert len(portfolio.transactions) == 1
    
    @patch('src.portfolio.StockDataFetcher')
    def test_buy_stock_insufficient_funds(self, mock_fetcher_class):
        """Test stock purchase with insufficient funds."""
        mock_fetcher = Mock()
        mock_fetcher.get_real_time_price.return_value = 1000.0
        mock_fetcher_class.return_value = mock_fetcher
        
        portfolio = Portfolio(initial_cash=5000.0)
        
        result = portfolio.buy_stock("AAPL", 10)  # Needs $10,000
        
        assert result is False
        assert portfolio.cash == 5000.0
        assert "AAPL" not in portfolio.holdings
        assert len(portfolio.transactions) == 0
    
    @patch('src.portfolio.StockDataFetcher')
    def test_sell_stock_success(self, mock_fetcher_class):
        """Test successful stock sale."""
        mock_fetcher = Mock()
        mock_fetcher.get_real_time_price.return_value = 110.0
        mock_fetcher_class.return_value = mock_fetcher
        
        portfolio = Portfolio(initial_cash=10000.0)
        
        # First buy some stock
        portfolio.holdings["AAPL"] = {"shares": 10, "avg_cost": 100.0}
        portfolio.cash = 9000.0
        
        result = portfolio.sell_stock("AAPL", 5)
        
        assert result is True
        assert portfolio.cash == 9550.0  # 9000 + (5 * 110)
        assert portfolio.holdings["AAPL"]["shares"] == 5
    
    def test_sell_stock_no_holdings(self):
        """Test selling stock with no holdings."""
        result = self.portfolio.sell_stock("AAPL", 10)
        
        assert result is False
        assert self.portfolio.cash == 10000.0
    
    def test_sell_stock_insufficient_shares(self):
        """Test selling more shares than owned."""
        self.portfolio.holdings["AAPL"] = {"shares": 5, "avg_cost": 100.0}
        
        result = self.portfolio.sell_stock("AAPL", 10)
        
        assert result is False
        assert self.portfolio.holdings["AAPL"]["shares"] == 5
    
    @patch('src.portfolio.StockDataFetcher')
    def test_get_portfolio_value(self, mock_fetcher_class):
        """Test portfolio value calculation."""
        mock_fetcher = Mock()
        mock_fetcher.get_real_time_price.side_effect = [110.0, 50.0]
        mock_fetcher_class.return_value = mock_fetcher
        
        portfolio = Portfolio(initial_cash=10000.0)
        portfolio.cash = 5000.0
        portfolio.holdings = {
            "AAPL": {"shares": 10, "avg_cost": 100.0},
            "GOOGL": {"shares": 20, "avg_cost": 45.0}
        }
        
        total_value = portfolio.get_portfolio_value()
        
        # 5000 cash + (10 * 110) + (20 * 50) = 5000 + 1100 + 1000 = 7100
        assert total_value == 7100.0

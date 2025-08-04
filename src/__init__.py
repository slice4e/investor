"""
Stock Investor Package

A comprehensive Python package for stock market analysis, portfolio management,
and investment strategy development.
"""

__version__ = "0.1.0"
__author__ = "Your Name"

from .data_fetcher import StockDataFetcher
from .portfolio import Portfolio
from .analyzer import StockAnalyzer
from .backtester import Backtester, CloseToOpenStrategy

__all__ = ["StockDataFetcher", "Portfolio", "StockAnalyzer", "Backtester", "CloseToOpenStrategy"]

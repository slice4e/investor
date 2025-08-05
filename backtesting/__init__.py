"""
Backtesting Framework Package

This package provides the infrastructure for backtesting investment strategies.

Modules:
- base_strategy: Abstract base class for all strategies
- strategy_manager: Manager for running and comparing strategies
"""

# Don't import strategy_manager here to avoid circular imports
from .base_strategy import BaseStrategy

__all__ = ['BaseStrategy']

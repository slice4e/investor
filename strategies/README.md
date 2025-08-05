# Investment Strategy Backtesting Framework

This directory contains a comprehensive backtesting framework for investment strategies with integrated data management.

## 🏗️ Architecture

### Core Components

1. **`backtesting/`** - Core backtesting infrastructure
   - `base_strategy.py` - Abstract base class for all strategies
   - `strategy_manager.py` - Manager for running and comparing strategies

2. **`strategies/`** - Strategy implementations
   - `buy_and_hold/` - Simple buy and hold strategy
   - *(Future strategies will be added here)*

## 📊 Available Strategies

### Buy and Hold Strategy
- **Location**: `strategies/buy_and_hold/buy_and_hold_strategy.py`
- **Description**: Simple long-term investment approach
- **Features**:
  - Buys maximum shares on start date
  - Holds position throughout investment period
  - Optionally sells at end date
  - Automatic data downloading if not available
  - Commission support
  - Comprehensive performance metrics

## 🚀 Quick Start

### 1. Run Individual Strategy

```python
from strategies.buy_and_hold.buy_and_hold_strategy import BuyAndHoldStrategy

# Initialize strategy
strategy = BuyAndHoldStrategy(
    initial_capital=10000.0,
    commission=0.0,
    sell_at_end=False
)

# Execute strategy
result = strategy.execute_strategy(
    ticker="AAPL",
    start_date="2020-01-01",
    end_date="2023-12-31"
)

# View results
print(f"Total Return: {result['performance']['total_return_pct']:.2f}%")
```

### 2. Use Strategy Manager

```python
from backtesting.strategy_manager import StrategyManager

# Initialize manager
manager = StrategyManager()

# Run strategy
result = manager.run_strategy(
    strategy_name='buy_and_hold',
    ticker='AAPL',
    start_date='2020-01-01',
    end_date='2023-12-31',
    strategy_params={'initial_capital': 10000.0}
)

# Compare strategies
comparison = manager.compare_strategies()
print(comparison)
```

### 3. Interactive Mode

```bash
python backtesting/strategy_manager.py
```

### 4. Run Tests

```bash
python test_strategies.py
```

## 📈 Strategy Parameters

### Buy and Hold Strategy Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `initial_capital` | float | 10000.0 | Starting capital in dollars |
| `commission` | float | 0.0 | Commission fee per trade |
| `sell_at_end` | bool | False | Whether to sell all positions at end date |

## 📊 Performance Metrics

All strategies provide comprehensive performance metrics:

- **Total Return (%)**: Overall return percentage
- **Annualized Return (%)**: Yearly average return
- **Volatility (%)**: Annual volatility (risk measure)
- **Sharpe Ratio**: Risk-adjusted return measure
- **Max Drawdown (%)**: Largest peak-to-trough decline
- **Win Rate (%)**: Percentage of profitable trades
- **Total Trades**: Number of buy/sell transactions

## 🔄 Data Integration

The framework automatically integrates with the data downloader:

1. **Automatic Downloads**: If historical data for a ticker is not available, it will be downloaded automatically
2. **Incremental Updates**: Uses the existing incremental update system
3. **Data Caching**: Caches data during strategy execution for performance
4. **Data Storage**: All downloaded data is stored in the `data/custom/` folder

## 🎯 Adding New Strategies

To add a new strategy:

1. Create a new folder in `strategies/` (e.g., `strategies/my_strategy/`)
2. Implement your strategy class inheriting from `BaseStrategy`
3. Implement required methods: `generate_signals()` and `execute_strategy()`
4. Register your strategy in the `StrategyManager`

### Example Strategy Template

```python
from backtesting.base_strategy import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self, initial_capital=10000.0):
        super().__init__("My Strategy", initial_capital)
    
    def generate_signals(self, data, ticker):
        # Implement your signal generation logic
        signals = data.copy()
        signals['signal'] = 'HOLD'  # Default to hold
        # Add your buy/sell logic here
        return signals
    
    def execute_strategy(self, ticker, start_date, end_date=None):
        # Implement your strategy execution logic
        # Use self.buy_stock() and self.sell_stock() for trades
        # Return results dictionary
        pass
```

## 📁 File Structure

```
strategies/
├── __init__.py
├── buy_and_hold/
│   ├── __init__.py
│   └── buy_and_hold_strategy.py
└── README.md

backtesting/
├── __init__.py
├── base_strategy.py
└── strategy_manager.py

test_strategies.py
```

## 🎪 Example Usage Scenarios

### Scenario 1: Test AAPL Buy & Hold for 3 Years
```python
strategy = BuyAndHoldStrategy(initial_capital=10000.0)
result = strategy.execute_strategy("AAPL", "2021-01-01", "2023-12-31")
```

### Scenario 2: Compare Multiple Tickers
```python
manager = StrategyManager()
tickers = ["AAPL", "MSFT", "GOOGL"]

for ticker in tickers:
    manager.run_strategy('buy_and_hold', ticker, '2020-01-01', '2023-12-31')

comparison = manager.compare_strategies()
```

### Scenario 3: Test with Commission Costs
```python
strategy = BuyAndHoldStrategy(
    initial_capital=10000.0,
    commission=9.99,  # $9.99 per trade
    sell_at_end=True
)
result = strategy.execute_strategy("SPY", "2020-01-01", "2023-12-31")
```

## 🔮 Future Enhancements

Planned strategy additions:
- Moving Average Crossover
- RSI-based Strategy
- Momentum Strategy
- Mean Reversion Strategy
- Dollar Cost Averaging
- Portfolio Rebalancing Strategies

## ⚠️ Important Notes

1. **Data Dependency**: Strategies require historical data. The framework will download missing data automatically.
2. **Date Handling**: All dates should be in 'YYYY-MM-DD' format or Python date objects.
3. **Commission Costs**: Commission is applied per trade (buy or sell).
4. **Performance Calculation**: Performance metrics are calculated based on daily portfolio values.
5. **Risk Disclaimer**: This is for educational and backtesting purposes only. Past performance does not guarantee future results.

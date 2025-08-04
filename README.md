# Stock Investor

A comprehensive Python application for stock market analysis, portfolio management, and investment strategy development.

## Features

### 📊 Data Fetching
- Real-time stock price data via Yahoo Finance
- Historical price data with customizable periods
- Stock information and company details
- Support for multiple data sources (extensible)

### 💼 Portfolio Management
- Buy and sell stock transactions
- Portfolio performance tracking
- Holdings summary and analysis
- Transaction history
- Real-time portfolio valuation

### 📈 Technical Analysis
- Moving averages (20, 50, 200 day)
- Relative Strength Index (RSI)
- Bollinger Bands
- MACD (Moving Average Convergence Divergence)
- Volatility calculations
- Comprehensive stock analysis charts

### 🖥️ Command Line Interface
- Interactive portfolio management
- Stock analysis commands
- Multi-stock comparison
- Visualization tools

## Installation

### Prerequisites
- Python 3.9 or higher
- pip (Python package installer)

### Setup

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd stock-investor
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment (optional)**
   ```bash
   copy .env.example .env
   # Edit .env file with your API keys if needed
   ```

## Usage

### Command Line Interface

The application provides several CLI commands:

#### Analyze a Stock
```bash
python -m src.main analyze AAPL --period 1y
```

#### Compare Multiple Stocks
```bash
python -m src.main compare AAPL GOOGL MSFT --period 6mo
```

#### Generate Stock Analysis Charts
```bash
python -m src.main plot AAPL --period 1y
```

#### Interactive Portfolio Management
```bash
python -m src.main portfolio
```

### Python API Usage

#### Data Fetching
```python
from src.data_fetcher import StockDataFetcher

fetcher = StockDataFetcher()
data = fetcher.get_stock_data("AAPL", period="1y")
info = fetcher.get_stock_info("AAPL")
price = fetcher.get_real_time_price("AAPL")
```

#### Portfolio Management
```python
from src.portfolio import Portfolio

portfolio = Portfolio(initial_cash=10000)
portfolio.buy_stock("AAPL", 10)  # Buy 10 shares
portfolio.sell_stock("AAPL", 5)  # Sell 5 shares

performance = portfolio.get_portfolio_performance()
holdings = portfolio.get_holdings_summary()
```

#### Stock Analysis
```python
from src.analyzer import StockAnalyzer

analyzer = StockAnalyzer()
analysis = analyzer.analyze_stock("AAPL", period="1y")
analyzer.plot_stock_analysis("AAPL")

# Compare multiple stocks
comparison = analyzer.compare_stocks(["AAPL", "GOOGL", "MSFT"])
```

## Project Structure

```
stock-investor/
├── src/                    # Source code
│   ├── __init__.py
│   ├── data_fetcher.py     # Stock data retrieval
│   ├── portfolio.py        # Portfolio management
│   ├── analyzer.py         # Technical analysis
│   └── main.py            # CLI application
├── tests/                  # Unit tests
│   ├── test_data_fetcher.py
│   └── test_portfolio.py
├── data/                   # Data storage
├── docs/                   # Documentation
├── requirements.txt        # Python dependencies
├── pyproject.toml         # Project configuration
├── .env.example           # Environment variables template
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/ tests/
```

### Type Checking
```bash
mypy src/
```

### Linting
```bash
flake8 src/ tests/
```

## Configuration

### Environment Variables
Create a `.env` file from `.env.example` and configure:

- `ALPHA_VANTAGE_API_KEY`: For Alpha Vantage data (optional)
- `LOG_LEVEL`: Logging level (INFO, DEBUG, WARNING, ERROR)
- `DATA_CACHE_ENABLED`: Enable/disable data caching
- `DATA_CACHE_DURATION_HOURS`: Cache duration in hours

### Data Sources
The application primarily uses Yahoo Finance (free) but can be extended with:
- Alpha Vantage (requires API key)
- Polygon.io (requires API key)
- IEX Cloud (requires API key)

## Features in Development

- [ ] Advanced portfolio optimization
- [ ] Risk analysis tools
- [ ] Backtesting framework
- [ ] Web dashboard interface
- [ ] Real-time alerts and notifications
- [ ] Options and derivatives support
- [ ] Machine learning price predictions

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Make your changes
4. Add tests for new functionality
5. Run the test suite (`pytest`)
6. Commit your changes (`git commit -am 'Add new feature'`)
7. Push to the branch (`git push origin feature/new-feature`)
8. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. It should not be used as the sole basis for investment decisions. Always consult with a qualified financial advisor before making investment decisions. The authors are not responsible for any financial losses incurred through the use of this software.

## Support

For questions, issues, or feature requests, please open an issue on the GitHub repository or contact the development team.

---

**Happy Investing! 📈**

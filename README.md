# Stock Investor 📈

A clean, professional Python foundation for stock market data analysis and investment strategy development. This project provides robust data downloading infrastructure as the foundation for building sophisticated investment strategies and backtesting frameworks.

## 🎯 Project Focus

This repository provides a **clean, focused foundation** for investment strategy development:

- **Core Data Infrastructure**: Professional-grade stock data downloading
- **Clean Architecture**: Minimal dependencies, maximum functionality  
- **Strategy Ready**: Built as a foundation for backtesting and analysis
- **Production Quality**: Robust error handling and corporate network support

## 🚀 Core Features

### 📊 Comprehensive Data Coverage
- **S&P 500**: Download all ~500 companies with maximum historical data available
- **NASDAQ**: Download NASDAQ-100 companies 
- **Custom Tickers**: Download any stock ticker symbols
- **Maximum History**: Goes back as far as data is available (some stocks to 1960s+)

### 📁 Multiple Export Formats  
- **CSV Files**: Individual files per stock or combined datasets
- **Excel Files**: Multi-sheet workbooks with summary information
- **Data Validation**: Automatic error handling and retry logic
- **Progress Tracking**: Real-time download progress and success rates

### ⚡ High Performance
- **Parallel Downloads**: Multi-threaded downloading for speed
- **Robust Error Handling**: Continues downloading even if some stocks fail
- **Memory Efficient**: Processes large datasets without memory issues
- **Corporate Network Support**: Works with proxy settings

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- Internet connection

### Setup
```bash
# Clone the repository
git clone https://github.com/slice4e/investor.git
cd investor

# Create virtual environment (recommended)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run the interactive downloader
python download_manager.py
```

## 📁 Project Structure

```
investor/
├── data_downloader.py      # Core StockDataDownloader class
├── download_manager.py     # Interactive CLI for data management
├── data/                   # Downloaded stock data storage
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── docs/                  # Additional documentation
```

## 🎯 Quick Start

### Interactive Mode (Recommended)
```bash
python download_manager.py
```
This launches an interactive menu with options for:
1. Download S&P 500 stocks
2. Download NASDAQ stocks  
3. Download custom tickers
4. Download complete market data
5. View downloaded files
6. Clean up data directory

### Programmatic Usage
```python
from data_downloader import StockDataDownloader

# Initialize downloader
downloader = StockDataDownloader(output_dir="my_data")

# Download specific stocks
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
data = downloader.download_custom_tickers(tickers)

# Save to Excel
downloader.save_to_excel(data, "tech_stocks.xlsx")

# Or save individual CSV files
downloader.save_to_csv(data, "individual_stocks")
```

## 📋 Data Fields

Each downloaded stock includes:
- **Date**: Trading date
- **Open**: Opening price
- **High**: Daily high price  
- **Low**: Daily low price
- **Close**: Closing price (adjusted)
- **Volume**: Trading volume
- **Ticker**: Stock symbol
- **Download_Date**: When data was downloaded

## 📊 Example Output

### Summary Statistics
```
📈 AAPL Analysis:
  Latest Price: $203.89
  52W High: $258.40
  52W Low: $172.19
  Records: 11,252 (1980-12-12 to 2025-08-05)
```

### File Structure
```
stock_market_data/
├── sp500/
│   ├── AAPL_historical_data.csv
│   ├── MSFT_historical_data.csv
│   └── ...
├── nasdaq/
│   ├── GOOGL_historical_data.csv
│   └── ...
├── custom/
│   └── custom_tickers_data.csv
├── sp500_data.xlsx
├── nasdaq_data.xlsx
└── download_report.csv
```

## 🔧 Configuration

### Environment Setup
```bash
# For corporate networks, set proxy (optional)
export http_proxy=http://proxy:port
export https_proxy=http://proxy:port
```

### Custom Output Directory
```python
# Use custom directory
downloader = StockDataDownloader(output_dir="custom_data_folder")
```

## 📈 Use Cases

### Investment Research
- Download complete S&P 500 for portfolio analysis
- Get historical data for fundamental analysis
- Research sector performance over time

### Backtesting Strategies
- Clean, validated data for strategy testing
- Maximum historical coverage for robust analysis
- Multiple tickers for diversification studies

### Academic Research
- Large datasets for financial modeling
- Long-term historical data for academic papers
- Clean, standardized data format

### Data Science Projects
- Feature engineering for ML models
- Time series analysis and forecasting
- Market pattern recognition

## 🛡️ Error Handling

The downloader includes robust error handling:
- **Network Issues**: Automatic retries for failed downloads
- **Invalid Tickers**: Continues with valid tickers, reports failures
- **Rate Limiting**: Respects API limits and includes delays
- **Data Validation**: Checks for empty or corrupt data

## 📊 Performance

### Benchmark Results
- **Single Stock**: ~1-2 seconds
- **10 Stocks**: ~5-10 seconds  
- **S&P 500 (~500 stocks)**: ~15-30 minutes
- **Complete Market**: ~45-60 minutes

Performance depends on:
- Internet connection speed
- Number of concurrent downloads (default: 10)
- Data history length per stock

## 🔮 Next Steps: Strategy Development

This clean foundation is ready for building investment strategies:

### Backtesting Framework
- Historical data analysis and strategy validation
- Performance metrics and risk assessment
- Portfolio optimization and rebalancing strategies

### Investment Strategies
- Technical analysis indicators and signals
- Fundamental analysis screening
- Quantitative trading algorithms
- Risk management systems

### Advanced Features  
- Real-time data integration
- Multi-asset portfolio management
- Machine learning model integration
- Strategy performance visualization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This tool is for educational and research purposes only. Not financial advice. 
Always verify data accuracy for production use.

## 🔗 Links

- **Repository**: https://github.com/slice4e/investor
- **Issues**: https://github.com/slice4e/investor/issues
- **Data Source**: Yahoo Finance (via yfinance library)

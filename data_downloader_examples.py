"""
Example Usage of Stock Data Downloader

This script demonstrates how to use the StockDataDownloader class
for various data downloading scenarios.
"""

from data_downloader import StockDataDownloader
import pandas as pd

def example_1_download_specific_stocks():
    """Example 1: Download specific stocks and save to Excel."""
    print("📊 Example 1: Downloading specific stocks")
    print("-" * 40)
    
    # Initialize downloader
    downloader = StockDataDownloader(output_dir="example_data")
    
    # Download specific stocks
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    data = downloader.download_custom_tickers(tickers)
    
    # Save to Excel
    downloader.save_to_excel(data, "tech_stocks.xlsx")
    
    # Print summary
    for ticker, df in data.items():
        if not df.empty:
            print(f"✅ {ticker}: {len(df)} records from {df['Date'].min()} to {df['Date'].max()}")

def example_2_download_sp500_sample():
    """Example 2: Download a sample of S&P 500 stocks."""
    print("\n📊 Example 2: Downloading S&P 500 sample")
    print("-" * 40)
    
    downloader = StockDataDownloader(output_dir="example_data")
    
    # Get S&P 500 tickers and take first 10
    sp500_tickers = downloader.get_sp500_tickers()[:10]
    print(f"Downloading sample of {len(sp500_tickers)} S&P 500 stocks...")
    
    data = downloader.download_multiple_stocks(sp500_tickers)
    
    # Create combined dataset
    combined_df = downloader.create_combined_dataset(data)
    
    if not combined_df.empty:
        # Save combined data
        combined_df.to_csv("example_data/sp500_sample.csv", index=False)
        
        # Show summary statistics
        print(f"\n📈 Combined dataset summary:")
        print(f"Total records: {len(combined_df):,}")
        print(f"Date range: {combined_df['Date'].min()} to {combined_df['Date'].max()}")
        print(f"Unique stocks: {combined_df['Ticker'].nunique()}")

def example_3_analyze_downloaded_data():
    """Example 3: Basic analysis of downloaded data."""
    print("\n📊 Example 3: Analyzing downloaded data")
    print("-" * 40)
    
    downloader = StockDataDownloader(output_dir="example_data")
    
    # Download a few stocks
    tickers = ['AAPL', 'MSFT', 'SPY']
    data = downloader.download_custom_tickers(tickers)
    
    for ticker, df in data.items():
        if not df.empty:
            # Calculate basic statistics
            latest_price = df['Close'].iloc[-1]
            price_52w_high = df['Close'].tail(252).max()  # Last 252 trading days ≈ 1 year
            price_52w_low = df['Close'].tail(252).min()
            
            # Calculate simple returns
            df['Daily_Return'] = df['Close'].pct_change()
            avg_daily_return = df['Daily_Return'].mean() * 100
            volatility = df['Daily_Return'].std() * 100
            
            print(f"\n📈 {ticker} Analysis:")
            print(f"  Latest Price: ${latest_price:.2f}")
            print(f"  52W High: ${price_52w_high:.2f}")
            print(f"  52W Low: ${price_52w_low:.2f}")
            print(f"  Avg Daily Return: {avg_daily_return:.3f}%")
            print(f"  Daily Volatility: {volatility:.3f}%")

def example_4_export_formats():
    """Example 4: Demonstrate different export formats."""
    print("\n📊 Example 4: Different export formats")
    print("-" * 40)
    
    downloader = StockDataDownloader(output_dir="example_data")
    
    # Download data
    tickers = ['AAPL', 'MSFT']
    data = downloader.download_custom_tickers(tickers)
    
    print("Saving data in different formats...")
    
    # 1. Individual CSV files
    downloader.save_to_csv(data, "individual_csv")
    print("✅ Individual CSV files saved")
    
    # 2. Excel file with multiple sheets
    downloader.save_to_excel(data, "multi_sheet_data.xlsx")
    print("✅ Multi-sheet Excel file saved")
    
    # 3. Combined CSV file
    combined_df = downloader.create_combined_dataset(data)
    if not combined_df.empty:
        combined_df.to_csv("example_data/combined_data.csv", index=False)
        print("✅ Combined CSV file saved")

def main():
    """Run all examples."""
    print("🚀 Stock Data Downloader Examples")
    print("=" * 50)
    
    try:
        example_1_download_specific_stocks()
        example_2_download_sp500_sample()
        example_3_analyze_downloaded_data()
        example_4_export_formats()
        
        print("\n✅ All examples completed successfully!")
        print("📁 Check the 'example_data' folder for output files")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")

if __name__ == "__main__":
    main()

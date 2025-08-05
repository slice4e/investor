"""
Quick Test of Data Downloader

Simple script to test the core functionality of downloading stock data.
"""

from data_downloader import StockDataDownloader

def quick_test():
    """Quick test of the data downloader."""
    print("🚀 Quick Data Downloader Test")
    print("=" * 40)
    
    # Initialize downloader
    downloader = StockDataDownloader(output_dir="test_data")
    
    # Test 1: Download a few popular stocks
    print("\n📊 Test 1: Downloading popular stocks")
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    data = downloader.download_custom_tickers(tickers)
    
    print(f"\n✅ Successfully downloaded {len(data)} stocks")
    
    # Show summary for each stock
    for ticker, df in data.items():
        if not df.empty:
            start_date = df['Date'].min().strftime('%Y-%m-%d')
            end_date = df['Date'].max().strftime('%Y-%m-%d')
            latest_price = df['Close'].iloc[-1]
            print(f"  📈 {ticker}: {len(df)} records ({start_date} to {end_date}) - Latest: ${latest_price:.2f}")
    
    # Test 2: Save to Excel (fixed timezone issue)
    print(f"\n📊 Test 2: Saving to Excel")
    try:
        downloader.save_to_excel(data, "test_stocks.xlsx")
        print("✅ Excel file saved successfully!")
    except Exception as e:
        print(f"❌ Excel save failed: {e}")
    
    # Test 3: Get S&P 500 tickers (now with lxml)
    print(f"\n📊 Test 3: Fetching S&P 500 tickers")
    try:
        sp500_tickers = downloader.get_sp500_tickers()
        print(f"✅ Found {len(sp500_tickers)} S&P 500 tickers")
        print(f"   First 10: {sp500_tickers[:10]}")
    except Exception as e:
        print(f"❌ S&P 500 fetch failed: {e}")
    
    # Generate report
    downloader.generate_download_report()

if __name__ == "__main__":
    quick_test()

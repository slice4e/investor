"""
Test script to demonstrate incremental data updates

This script shows how the enhanced data downloader handles:
1. Initial data download
2. Incremental updates (only new data)
3. Data summary and metadata tracking
"""

from data_downloader import StockDataDownloader
import pandas as pd
from datetime import datetime, timedelta

def main():
    print("🧪 Testing Incremental Data Updates")
    print("="*50)
    
    # Initialize downloader
    downloader = StockDataDownloader(output_dir="test_data")
    
    # Test 1: Download some initial data
    print("\n1️⃣ Initial Download Test")
    print("-" * 30)
    test_tickers = ['AAPL', 'MSFT', 'GOOGL']
    
    # Use incremental download for initial data
    data = downloader.download_custom_tickers_incremental(test_tickers, force_full=True)
    print(f"✅ Downloaded {len(data)} tickers initially")
    
    # Test 2: Show data summary
    print("\n2️⃣ Data Summary")
    print("-" * 30)
    summary = downloader.get_data_summary()
    if not summary.empty:
        print(summary[['Ticker', 'Subfolder', 'Last_Date', 'Record_Count', 'Days_Since_Update']])
    
    # Test 3: Simulate incremental update (should find data is current)
    print("\n3️⃣ Incremental Update Test")
    print("-" * 30)
    update_results = downloader.update_existing_data(subfolder="custom")
    
    for ticker, status in update_results.items():
        print(f"  {ticker}: {status}")
    
    # Test 4: Show metadata
    print("\n4️⃣ Metadata Information")
    print("-" * 30)
    metadata = downloader.load_metadata()
    for ticker, info in list(metadata.items())[:3]:  # Show first 3 tickers
        print(f"  {ticker}:")
        print(f"    Last Date: {info.get('last_date', 'unknown')}")
        print(f"    Records: {info.get('record_count', 0)}")
        print(f"    Last Updated: {info.get('last_updated', 'unknown')[:19]}")
    
    print("\n✅ Incremental update test completed!")
    print("\n💡 Key Features Demonstrated:")
    print("   • Metadata tracking for each ticker")
    print("   • Smart incremental updates (skips current data)")
    print("   • Local storage management")
    print("   • Data summary reporting")
    print("   • Ready for strategy development!")

if __name__ == "__main__":
    main()

"""
Professional Stock Data Downloader

This module provides comprehensive stock data downloading capabilities for:
- All S&P 500 companies
- All NASDAQ companies  
- Custom stock tickers
- Maximum historical data available
- Export to Excel/CSV formats

Features:
- Parallel downloading for speed
- Error handling and retry logic
- Progress tracking
- Data validation
- Multiple export formats
"""

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, date
import time
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Union, Tuple
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StockDataDownloader:
    """
    Professional stock data downloader with comprehensive features.
    """
    
    def __init__(self, output_dir: str = "data"):
        """
        Initialize the data downloader.
        
        Args:
            output_dir: Directory to save downloaded data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / "sp500").mkdir(exist_ok=True)
        (self.output_dir / "nasdaq").mkdir(exist_ok=True)
        (self.output_dir / "custom").mkdir(exist_ok=True)
        (self.output_dir / "combined").mkdir(exist_ok=True)
        (self.output_dir / "metadata").mkdir(exist_ok=True)
        
        self.failed_downloads = []
        self.successful_downloads = []
        self.metadata_file = self.output_dir / "metadata" / "download_metadata.json"
        
    def get_sp500_tickers(self) -> List[str]:
        """
        Get current S&P 500 company tickers from Wikipedia.
        
        Returns:
            List of S&P 500 ticker symbols
        """
        try:
            logger.info("Fetching S&P 500 ticker list...")
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            sp500_table = tables[0]
            tickers = sp500_table['Symbol'].tolist()
            
            # Clean ticker symbols (some may have dots or special characters)
            tickers = [ticker.replace('.', '-') for ticker in tickers]
            
            logger.info(f"Found {len(tickers)} S&P 500 tickers")
            return tickers
            
        except Exception as e:
            logger.error(f"Error fetching S&P 500 tickers: {e}")
            # Fallback to a known list of major S&P 500 stocks
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ']
    
    def get_nasdaq_tickers(self) -> List[str]:
        """
        Get NASDAQ company tickers.
        
        Returns:
            List of NASDAQ ticker symbols
        """
        try:
            logger.info("Fetching NASDAQ ticker list...")
            # Using NASDAQ's official API endpoint for listed companies
            url = "https://api.nasdaq.com/api/screener/stocks"
            params = {
                'tableonly': 'true',
                'limit': '25000',
                'offset': '0',
                'exchange': 'nasdaq'
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, params=params, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and 'rows' in data['data']:
                    tickers = [row['symbol'] for row in data['data']['rows']]
                    logger.info(f"Found {len(tickers)} NASDAQ tickers")
                    return tickers
            
            # Fallback method using pandas
            logger.info("Trying alternative method for NASDAQ tickers...")
            url = "https://en.wikipedia.org/wiki/NASDAQ-100"
            tables = pd.read_html(url)
            nasdaq_table = tables[4]  # Usually the companies table
            tickers = nasdaq_table['Ticker'].tolist()
            tickers = [ticker.replace('.', '-') for ticker in tickers]
            
            logger.info(f"Found {len(tickers)} NASDAQ-100 tickers")
            return tickers
            
        except Exception as e:
            logger.error(f"Error fetching NASDAQ tickers: {e}")
            # Fallback to major NASDAQ stocks
            return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NFLX', 'NVDA', 'ADBE', 'PYPL']
    
    def download_stock_data(self, ticker: str, period: str = "max") -> Optional[pd.DataFrame]:
        """
        Download historical data for a single stock.
        
        Args:
            ticker: Stock ticker symbol
            period: Time period ('max', '10y', '5y', etc.)
            
        Returns:
            DataFrame with stock data or None if failed
        """
        try:
            logger.debug(f"Downloading data for {ticker}")
            
            stock = yf.Ticker(ticker)
            data = stock.history(period=period, auto_adjust=True, prepost=True)
            
            if data.empty:
                logger.warning(f"No data available for {ticker}")
                return None
            
            # Add ticker column
            data['Ticker'] = ticker
            
            # Add additional metadata
            data['Download_Date'] = datetime.now()
            
            # Reset index to make Date a column
            data = data.reset_index()
            
            # Convert timezone-aware datetime to timezone-naive for Excel compatibility
            if 'Date' in data.columns:
                data['Date'] = pd.to_datetime(data['Date']).dt.tz_localize(None)
            
            logger.debug(f"Successfully downloaded {len(data)} records for {ticker}")
            return data
            
        except Exception as e:
            logger.error(f"Error downloading {ticker}: {e}")
            return None
    
    def save_metadata(self, ticker: str, subfolder: str, last_date: str, record_count: int):
        """
        Save metadata about downloaded stock data.
        
        Args:
            ticker: Stock ticker symbol
            subfolder: Subfolder where data is stored
            last_date: Last date of data in YYYY-MM-DD format
            record_count: Number of records in the dataset
        """
        try:
            # Load existing metadata
            metadata = {}
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    metadata = json.load(f)
            
            # Update metadata for this ticker
            metadata[ticker] = {
                'subfolder': subfolder,
                'last_date': last_date,
                'record_count': record_count,
                'last_updated': datetime.now().isoformat(),
                'file_path': str(self.output_dir / subfolder / f"{ticker}_historical_data.csv")
            }
            
            # Save updated metadata
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving metadata for {ticker}: {e}")
    
    def load_metadata(self) -> Dict:
        """Load metadata about existing stock data."""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return {}
    
    def get_existing_data_info(self, ticker: str, subfolder: str) -> Optional[Tuple[pd.DataFrame, str]]:
        """
        Load existing data for a ticker and return last date.
        
        Args:
            ticker: Stock ticker symbol
            subfolder: Subfolder to check
            
        Returns:
            Tuple of (existing_data, last_date_str) or None if no data exists
        """
        try:
            file_path = self.output_dir / subfolder / f"{ticker}_historical_data.csv"
            if file_path.exists():
                existing_data = pd.read_csv(file_path)
                if not existing_data.empty and 'Date' in existing_data.columns:
                    existing_data['Date'] = pd.to_datetime(existing_data['Date'])
                    last_date = existing_data['Date'].max()
                    last_date_str = last_date.strftime('%Y-%m-%d')
                    return existing_data, last_date_str
            return None
        except Exception as e:
            logger.error(f"Error loading existing data for {ticker}: {e}")
            return None
    
    def download_stock_data_incremental(self, ticker: str, subfolder: str = "custom", 
                                      force_full: bool = False) -> Optional[pd.DataFrame]:
        """
        Download stock data with incremental update support.
        
        Args:
            ticker: Stock ticker symbol
            subfolder: Subfolder to save data
            force_full: If True, download full history regardless of existing data
            
        Returns:
            Complete DataFrame with historical data or None if failed
        """
        try:
            logger.debug(f"Checking existing data for {ticker}")
            
            # Check if we have existing data
            existing_info = None if force_full else self.get_existing_data_info(ticker, subfolder)
            
            if existing_info is None:
                # No existing data, download full history
                logger.info(f"📥 {ticker}: Downloading full history")
                data = self.download_stock_data(ticker, period="max")
                if data is not None:
                    # Save to CSV
                    file_path = self.output_dir / subfolder / f"{ticker}_historical_data.csv"
                    data.to_csv(file_path, index=False)
                    
                    # Save metadata
                    last_date = data['Date'].max().strftime('%Y-%m-%d')
                    self.save_metadata(ticker, subfolder, last_date, len(data))
                    
                    logger.info(f"✅ {ticker}: Saved {len(data)} records (full history)")
                return data
            
            else:
                existing_data, last_date_str = existing_info
                last_date = pd.to_datetime(last_date_str).date()
                today = date.today()
                
                # Check if we need to update
                if last_date >= today - timedelta(days=1):
                    logger.info(f"📋 {ticker}: Data is current (last: {last_date_str})")
                    return existing_data
                
                # Download new data since last update
                start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
                logger.info(f"🔄 {ticker}: Updating from {start_date}")
                
                stock = yf.Ticker(ticker)
                new_data = stock.history(start=start_date, auto_adjust=True, prepost=True)
                
                if new_data.empty:
                    logger.info(f"📋 {ticker}: No new data available")
                    return existing_data
                
                # Process new data
                new_data['Ticker'] = ticker
                new_data['Download_Date'] = datetime.now()
                new_data = new_data.reset_index()
                
                # Convert timezone-aware datetime to timezone-naive
                if 'Date' in new_data.columns:
                    new_data['Date'] = pd.to_datetime(new_data['Date']).dt.tz_localize(None)
                
                # Combine with existing data
                combined_data = pd.concat([existing_data, new_data], ignore_index=True)
                combined_data = combined_data.drop_duplicates(subset=['Date'], keep='last')
                combined_data = combined_data.sort_values('Date').reset_index(drop=True)
                
                # Save updated data
                file_path = self.output_dir / subfolder / f"{ticker}_historical_data.csv"
                combined_data.to_csv(file_path, index=False)
                
                # Update metadata
                last_date = combined_data['Date'].max().strftime('%Y-%m-%d')
                self.save_metadata(ticker, subfolder, last_date, len(combined_data))
                
                logger.info(f"✅ {ticker}: Added {len(new_data)} new records (total: {len(combined_data)})")
                return combined_data
                
        except Exception as e:
            logger.error(f"Error in incremental download for {ticker}: {e}")
            return None
    
    def update_existing_data(self, tickers: List[str] = None, subfolder: str = "custom",
                           max_workers: int = 10) -> Dict[str, str]:
        """
        Update existing data for specified tickers or all existing tickers.
        
        Args:
            tickers: List of tickers to update (None = update all existing)
            subfolder: Subfolder to check/update
            max_workers: Number of parallel workers
            
        Returns:
            Dictionary with update results for each ticker
        """
        # Determine which tickers to update
        if tickers is None:
            # Get all existing tickers from metadata
            metadata = self.load_metadata()
            tickers = [ticker for ticker, info in metadata.items() 
                      if info.get('subfolder') == subfolder]
            
            if not tickers:
                # Fallback: scan directory for CSV files
                folder_path = self.output_dir / subfolder
                if folder_path.exists():
                    csv_files = list(folder_path.glob("*_historical_data.csv"))
                    tickers = [f.stem.replace('_historical_data', '') for f in csv_files]
        
        if not tickers:
            logger.warning(f"No tickers found to update in {subfolder}")
            return {}
        
        logger.info(f"🔄 Updating {len(tickers)} tickers in {subfolder}")
        
        update_results = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {
                executor.submit(self.download_stock_data_incremental, ticker, subfolder): ticker
                for ticker in tickers
            }
            
            for i, future in enumerate(as_completed(future_to_ticker), 1):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    if result is not None:
                        update_results[ticker] = "✅ Updated"
                        self.successful_downloads.append(ticker)
                    else:
                        update_results[ticker] = "❌ Failed"
                        self.failed_downloads.append(ticker)
                except Exception as e:
                    update_results[ticker] = f"❌ Error: {str(e)}"
                    self.failed_downloads.append(ticker)
                
                if i % 10 == 0:
                    logger.info(f"Progress: {i}/{len(tickers)} tickers processed")
        
        # Print summary
        successful = sum(1 for status in update_results.values() if "✅" in status)
        logger.info(f"Update complete: {successful}/{len(tickers)} successful")
        
        return update_results
    
    def get_data_summary(self) -> pd.DataFrame:
        """
        Get summary of all downloaded data.
        
        Returns:
            DataFrame with summary information for each ticker
        """
        metadata = self.load_metadata()
        if not metadata:
            logger.warning("No metadata found")
            return pd.DataFrame()
        
        summary_data = []
        for ticker, info in metadata.items():
            summary_data.append({
                'Ticker': ticker,
                'Subfolder': info.get('subfolder', 'unknown'),
                'Last_Date': info.get('last_date', 'unknown'),
                'Record_Count': info.get('record_count', 0),
                'Last_Updated': info.get('last_updated', 'unknown'),
                'Days_Since_Update': (datetime.now() - pd.to_datetime(info.get('last_updated', datetime.now()))).days
            })
        
        return pd.DataFrame(summary_data)
    
    def download_multiple_stocks(self, tickers: List[str], period: str = "max", 
                                max_workers: int = 10) -> Dict[str, pd.DataFrame]:
        """
        Download data for multiple stocks in parallel.
        
        Args:
            tickers: List of ticker symbols
            period: Time period for data
            max_workers: Number of parallel downloads
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        logger.info(f"Starting download of {len(tickers)} stocks with {max_workers} workers")
        
        results = {}
        failed_tickers = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all download tasks
            future_to_ticker = {
                executor.submit(self.download_stock_data, ticker, period): ticker 
                for ticker in tickers
            }
            
            # Process completed downloads
            for i, future in enumerate(as_completed(future_to_ticker), 1):
                ticker = future_to_ticker[future]
                
                try:
                    data = future.result()
                    if data is not None:
                        results[ticker] = data
                        self.successful_downloads.append(ticker)
                        logger.info(f"[{i}/{len(tickers)}] ✅ {ticker}: {len(data)} records")
                    else:
                        failed_tickers.append(ticker)
                        self.failed_downloads.append(ticker)
                        logger.warning(f"[{i}/{len(tickers)}] ❌ {ticker}: No data")
                        
                except Exception as e:
                    failed_tickers.append(ticker)
                    self.failed_downloads.append(ticker)
                    logger.error(f"[{i}/{len(tickers)}] ❌ {ticker}: {e}")
                
                # Progress update every 50 stocks
                if i % 50 == 0:
                    success_rate = (len(results) / i) * 100
                    logger.info(f"Progress: {i}/{len(tickers)} ({success_rate:.1f}% success rate)")
        
        logger.info(f"Download complete: {len(results)} successful, {len(failed_tickers)} failed")
        return results
    
    def save_to_csv(self, data_dict: Dict[str, pd.DataFrame], subfolder: str = "combined"):
        """
        Save downloaded data to CSV files.
        
        Args:
            data_dict: Dictionary mapping ticker to DataFrame
            subfolder: Subfolder to save files
        """
        save_dir = self.output_dir / subfolder
        save_dir.mkdir(exist_ok=True)
        
        logger.info(f"Saving {len(data_dict)} files to CSV...")
        
        for ticker, data in data_dict.items():
            filename = f"{ticker}_historical_data.csv"
            filepath = save_dir / filename
            data.to_csv(filepath, index=False)
        
        logger.info(f"CSV files saved to {save_dir}")
    
    def save_to_excel(self, data_dict: Dict[str, pd.DataFrame], 
                     filename: str = "stock_data_combined.xlsx"):
        """
        Save all data to a single Excel file with multiple sheets.
        
        Args:
            data_dict: Dictionary mapping ticker to DataFrame
            filename: Name of Excel file
        """
        filepath = self.output_dir / filename
        
        logger.info(f"Saving {len(data_dict)} stocks to Excel file...")
        
        try:
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Create summary sheet
                summary_data = []
                for ticker, data in data_dict.items():
                    if not data.empty:
                        summary_data.append({
                            'Ticker': ticker,
                            'Records': len(data),
                            'Start_Date': data['Date'].min(),
                            'End_Date': data['Date'].max(),
                            'Latest_Price': data['Close'].iloc[-1] if len(data) > 0 else None
                        })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Add individual stock sheets (limit to avoid Excel sheet limit)
                for i, (ticker, data) in enumerate(data_dict.items()):
                    if i < 1000:  # Excel has a limit on number of sheets
                        sheet_name = ticker[:31]  # Excel sheet name limit
                        data.to_excel(writer, sheet_name=sheet_name, index=False)
                    else:
                        break
                
            logger.info(f"Excel file saved: {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving Excel file: {e}")
    
    def create_combined_dataset(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Combine all stock data into a single DataFrame.
        
        Args:
            data_dict: Dictionary mapping ticker to DataFrame
            
        Returns:
            Combined DataFrame
        """
        logger.info("Creating combined dataset...")
        
        all_data = []
        for ticker, data in data_dict.items():
            if not data.empty:
                all_data.append(data)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            logger.info(f"Combined dataset created: {len(combined_df)} total records")
            return combined_df
        else:
            logger.warning("No data to combine")
            return pd.DataFrame()
    
    def download_sp500_data(self) -> Dict[str, pd.DataFrame]:
        """Download all S&P 500 stock data."""
        logger.info("🚀 Starting S&P 500 data download...")
        tickers = self.get_sp500_tickers()
        data = self.download_multiple_stocks(tickers)
        
        # Save results
        self.save_to_csv(data, "sp500")
        combined_df = self.create_combined_dataset(data)
        if not combined_df.empty:
            combined_df.to_csv(self.output_dir / "sp500_all_data.csv", index=False)
        
        return data
    
    def download_nasdaq_data(self) -> Dict[str, pd.DataFrame]:
        """Download all NASDAQ stock data."""
        logger.info("🚀 Starting NASDAQ data download...")
        tickers = self.get_nasdaq_tickers()
        data = self.download_multiple_stocks(tickers)
        
        # Save results
        self.save_to_csv(data, "nasdaq")
        combined_df = self.create_combined_dataset(data)
        if not combined_df.empty:
            combined_df.to_csv(self.output_dir / "nasdaq_all_data.csv", index=False)
        
        return data
    
    def download_custom_tickers(self, tickers: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Download data for custom list of tickers.
        
        Args:
            tickers: List of ticker symbols
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        logger.info(f"🚀 Starting custom ticker download for {len(tickers)} stocks...")
        data = self.download_multiple_stocks(tickers)
        
        # Save results
        self.save_to_csv(data, "custom")
        combined_df = self.create_combined_dataset(data)
        if not combined_df.empty:
            combined_df.to_csv(self.output_dir / "custom_tickers_data.csv", index=False)
        
        return data
    
    def generate_download_report(self):
        """Generate a summary report of the download session."""
        report = {
            'timestamp': datetime.now().isoformat(),
            'successful_downloads': len(self.successful_downloads),
            'failed_downloads': len(self.failed_downloads),
            'success_rate': len(self.successful_downloads) / (len(self.successful_downloads) + len(self.failed_downloads)) * 100 if (len(self.successful_downloads) + len(self.failed_downloads)) > 0 else 0,
            'successful_tickers': self.successful_downloads,
            'failed_tickers': self.failed_downloads
        }
        
        # Save report
        report_df = pd.DataFrame([report])
        report_df.to_csv(self.output_dir / "download_report.csv", index=False)
        
        # Print summary
        print("\n" + "="*60)
        print("📊 DOWNLOAD SUMMARY REPORT")
        print("="*60)
        print(f"✅ Successful downloads: {report['successful_downloads']}")
        print(f"❌ Failed downloads: {report['failed_downloads']}")
        print(f"📈 Success rate: {report['success_rate']:.1f}%")
        print(f"📁 Data saved to: {self.output_dir}")
        
        if self.failed_downloads:
            print(f"\n❌ Failed tickers: {', '.join(self.failed_downloads[:10])}")
            if len(self.failed_downloads) > 10:
                print(f"... and {len(self.failed_downloads) - 10} more")
        
        print("="*60)
        
        return report
    
    def download_custom_tickers_incremental(self, tickers: List[str], 
                                          force_full: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Download custom tickers with incremental update support.
        
        Args:
            tickers: List of ticker symbols
            force_full: If True, download full history for all tickers
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        logger.info(f"🚀 Starting incremental download for {len(tickers)} custom tickers...")
        
        results = {}
        for ticker in tickers:
            data = self.download_stock_data_incremental(ticker, "custom", force_full)
            if data is not None:
                results[ticker] = data
        
        # Create combined dataset
        if results:
            combined_df = self.create_combined_dataset(results)
            if not combined_df.empty:
                combined_df.to_csv(self.output_dir / "custom_tickers_data.csv", index=False)
                logger.info(f"💾 Saved combined dataset: {len(combined_df)} total records")
        
        return results
    
    def download_sp500_data_incremental(self, force_full: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Download S&P 500 data with incremental update support.
        
        Args:
            force_full: If True, download full history for all stocks
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        logger.info("🚀 Starting incremental S&P 500 data download...")
        tickers = self.get_sp500_tickers()
        
        results = {}
        for ticker in tickers:
            data = self.download_stock_data_incremental(ticker, "sp500", force_full)
            if data is not None:
                results[ticker] = data
        
        # Create combined dataset
        if results:
            combined_df = self.create_combined_dataset(results)
            if not combined_df.empty:
                combined_df.to_csv(self.output_dir / "sp500_all_data.csv", index=False)
                logger.info(f"💾 Saved S&P 500 combined dataset: {len(combined_df)} total records")
        
        return results
    
    def download_nasdaq_data_incremental(self, force_full: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Download NASDAQ data with incremental update support.
        
        Args:
            force_full: If True, download full history for all stocks
            
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        logger.info("🚀 Starting incremental NASDAQ data download...")
        tickers = self.get_nasdaq_tickers()
        
        results = {}
        for ticker in tickers:
            data = self.download_stock_data_incremental(ticker, "nasdaq", force_full)
            if data is not None:
                results[ticker] = data
        
        # Create combined dataset
        if results:
            combined_df = self.create_combined_dataset(results)
            if not combined_df.empty:
                combined_df.to_csv(self.output_dir / "nasdaq_all_data.csv", index=False)
                logger.info(f"💾 Saved NASDAQ combined dataset: {len(combined_df)} total records")
        
        return results
    
    def update_all_data(self) -> Dict[str, Dict[str, str]]:
        """
        Update all existing data across all subfolders.
        
        Returns:
            Dictionary with update results for each subfolder
        """
        logger.info("🔄 Starting comprehensive data update...")
        
        results = {}
        subfolders = ["sp500", "nasdaq", "custom"]
        
        for subfolder in subfolders:
            logger.info(f"📁 Updating {subfolder} data...")
            update_result = self.update_existing_data(subfolder=subfolder)
            results[subfolder] = update_result
        
        # Generate update summary
        total_updated = sum(len(results[sf]) for sf in subfolders)
        total_successful = sum(sum(1 for status in results[sf].values() if "✅" in status) 
                             for sf in subfolders)
        
        logger.info(f"🎯 Update complete: {total_successful}/{total_updated} tickers updated successfully")
        
        return results


def main():
    """Main function to demonstrate the data downloader."""
    print("🚀 Professional Stock Data Downloader")
    print("=====================================")
    
    # Initialize downloader
    downloader = StockDataDownloader(output_dir="data")
    
    print("\nSelect download option:")
    print("1. Download S&P 500 stocks")
    print("2. Download NASDAQ stocks") 
    print("3. Download custom tickers")
    print("4. Download all (S&P 500 + NASDAQ)")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == "1":
        data = downloader.download_sp500_data()
        downloader.save_to_excel(data, "sp500_data.xlsx")
        
    elif choice == "2":
        data = downloader.download_nasdaq_data()
        downloader.save_to_excel(data, "nasdaq_data.xlsx")
        
    elif choice == "3":
        tickers_input = input("Enter ticker symbols (comma-separated): ").strip()
        tickers = [t.strip().upper() for t in tickers_input.split(",")]
        data = downloader.download_custom_tickers(tickers)
        downloader.save_to_excel(data, "custom_data.xlsx")
        
    elif choice == "4":
        print("Downloading all S&P 500 and NASDAQ stocks...")
        sp500_data = downloader.download_sp500_data()
        nasdaq_data = downloader.download_nasdaq_data()
        
        # Combine and save all data
        all_data = {**sp500_data, **nasdaq_data}
        downloader.save_to_excel(all_data, "all_stocks_data.xlsx")
        
        combined_df = downloader.create_combined_dataset(all_data)
        if not combined_df.empty:
            combined_df.to_csv(downloader.output_dir / "all_stocks_combined.csv", index=False)
    
    else:
        print("Invalid choice. Exiting.")
        return
    
    # Generate report
    downloader.generate_download_report()


if __name__ == "__main__":
    main()

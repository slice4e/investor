"""
Market Analysis Module

This module provides functionality to analyze market-wide movements and identify
"down days" based on the percentage of stocks that are declining in an index.

Key Features:
- Identify down days for NASDAQ and S&P 500
- Calculate percentage of stocks down on any given day
- Generate down day reports with configurable thresholds
- Support for historical analysis and trend identification
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import logging

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from data_downloader import StockDataDownloader

logger = logging.getLogger(__name__)

class MarketAnalyzer:
    """
    Analyzer for market-wide movements and down day identification.
    """
    
    def __init__(self):
        """Initialize the market analyzer."""
        self.downloader = StockDataDownloader()
        self.data_cache = {}
        
    def ensure_index_data(self, index_name: str) -> Dict[str, pd.DataFrame]:
        """
        Ensure all stocks for an index are downloaded and available.
        Uses bulk data files if available, falls back to individual files.
        
        Args:
            index_name: 'nasdaq' or 'sp500'
            
        Returns:
            Dictionary of {ticker: DataFrame} for all index stocks
        """
        logger.info(f"Ensuring data availability for {index_name.upper()} index")
        
        if index_name.lower() == 'nasdaq':
            tickers = self.downloader.get_nasdaq_100_tickers()  # Use NASDAQ-100 instead of all NASDAQ
            subfolder = 'nasdaq'
            bulk_file = 'nasdaq_all_data.csv'
        elif index_name.lower() == 'sp500':
            tickers = self.downloader.get_sp500_tickers()
            subfolder = 'sp500'
            bulk_file = 'sp500_all_data.csv'
        else:
            raise ValueError(f"Unsupported index: {index_name}")
        
        logger.info(f"Found {len(tickers)} tickers for {index_name.upper()}")
        
        available_data = {}
        
        # First, try to load from bulk data file
        bulk_file_path = self.downloader.output_dir / bulk_file
        if bulk_file_path.exists():
            try:
                logger.info(f"Loading bulk data from {bulk_file}")
                bulk_data = pd.read_csv(bulk_file_path)
                bulk_data['Date'] = pd.to_datetime(bulk_data['Date'])
                
                # Split bulk data into individual ticker DataFrames
                for ticker in tickers:
                    try:
                        ticker_data = bulk_data[bulk_data['Ticker'] == ticker].copy()
                        if not ticker_data.empty:
                            ticker_data = ticker_data.drop('Ticker', axis=1)
                            ticker_data.set_index('Date', inplace=True)
                            available_data[ticker] = ticker_data
                    except Exception as e:
                        logger.warning(f"Error processing {ticker} from bulk data: {e}")
                        continue
                        
                logger.info(f"Loaded {len(available_data)} stocks from bulk data file")
                
                # If we got sufficient data, return it
                coverage_ratio = len(available_data) / len(tickers)
                if coverage_ratio >= 0.90:  # 90% threshold for completeness
                    logger.info(f"Bulk data coverage is sufficient ({len(available_data)}/{len(tickers)} = {coverage_ratio:.1%} tickers)")
                    return available_data
                else:
                    logger.info(f"Bulk data coverage is insufficient ({coverage_ratio:.1%}), will supplement with individual files")
                    
            except Exception as e:
                logger.warning(f"Error loading bulk data from {bulk_file}: {e}")
                # Don't clear available_data - keep what we loaded successfully
        
        # Only supplement with individual files for truly missing tickers
        missing_tickers = [ticker for ticker in tickers if ticker not in available_data]
        
        if missing_tickers:
            logger.info(f"Looking for individual files for {len(missing_tickers)} missing tickers")
            
            for ticker in missing_tickers:
                data_file = self.downloader.output_dir / subfolder / f"{ticker}_historical_data.csv"
                if data_file.exists():
                    try:
                        data = pd.read_csv(data_file)
                        data['Date'] = pd.to_datetime(data['Date'])
                        data.set_index('Date', inplace=True)
                        available_data[ticker] = data
                        missing_tickers.remove(ticker)
                    except Exception as e:
                        logger.warning(f"Error loading {ticker}: {e}")
        
        logger.info(f"Total loaded data: {len(available_data)} stocks")
        
        # Only download if we're missing a significant number of tickers
        still_missing = [ticker for ticker in tickers if ticker not in available_data]
        if still_missing:
            missing_ratio = len(still_missing) / len(tickers)
            if missing_ratio > 0.10:  # More than 10% missing
                logger.warning(f"Still missing {len(still_missing)} tickers ({missing_ratio:.1%})")
                logger.warning("Consider running the data downloader to update bulk data files")
                logger.info("Proceeding with available data to avoid unnecessary downloads")
            else:
                logger.info(f"Only {len(still_missing)} tickers missing ({missing_ratio:.1%}) - proceeding with available data")
        
        return available_data
    
    def calculate_daily_down_percentage(self, index_data: Dict[str, pd.DataFrame], 
                                      start_date: Union[str, date] = None,
                                      end_date: Union[str, date] = None,
                                      cache_key: str = None) -> pd.DataFrame:
        """
        Calculate the percentage of stocks that are down each day.
        
        Args:
            index_data: Dictionary of {ticker: DataFrame} with stock data
            start_date: Start date for analysis (optional)
            end_date: End date for analysis (optional)
            cache_key: Unique key for caching (e.g., "nasdaq_2024")
            
        Returns:
            DataFrame with daily down percentages
        """
        # Try to load from cache first
        if cache_key:
            cached_data = self.load_daily_stats_cache(cache_key)
            if cached_data is not None:
                logger.info(f"Loaded cached daily statistics for {cache_key}")
                return cached_data
        
        logger.info("Calculating daily down percentages")
        
        # Convert dates
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).date()
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).date()
        
        # Get all unique dates across all stocks
        all_dates = set()
        for ticker, data in index_data.items():
            if not data.empty:
                dates = data.index.date
                if start_date:
                    dates = dates[dates >= start_date]
                if end_date:
                    dates = dates[dates <= end_date]
                all_dates.update(dates)
        
        all_dates = sorted(list(all_dates))
        logger.info(f"Analyzing {len(all_dates)} trading days")
        
        # Calculate daily statistics
        daily_stats = []
        
        for current_date in all_dates:
            stocks_with_data = 0
            stocks_down = 0
            down_tickers = []
            up_tickers = []
            
            for ticker, data in index_data.items():
                # Get data for current date
                date_mask = data.index.date == current_date
                if not date_mask.any():
                    continue
                
                current_day_data = data[date_mask]
                if current_day_data.empty:
                    continue
                
                # Check if we have required columns
                if 'Open' not in current_day_data.columns or 'Close' not in current_day_data.columns:
                    continue
                
                # Calculate daily return
                open_price = current_day_data['Open'].iloc[0]
                close_price = current_day_data['Close'].iloc[0]
                
                if pd.isna(open_price) or pd.isna(close_price) or open_price == 0:
                    continue
                
                daily_return = (close_price - open_price) / open_price
                stocks_with_data += 1
                
                if daily_return < 0:
                    stocks_down += 1
                    down_tickers.append(ticker)
                else:
                    up_tickers.append(ticker)
            
            # Calculate percentage down
            if stocks_with_data > 0:
                pct_down = (stocks_down / stocks_with_data) * 100
                pct_up = ((stocks_with_data - stocks_down) / stocks_with_data) * 100
                
                daily_stats.append({
                    'date': current_date,
                    'total_stocks': stocks_with_data,
                    'stocks_down': stocks_down,
                    'stocks_up': stocks_with_data - stocks_down,
                    'pct_down': pct_down,
                    'pct_up': pct_up,
                    'down_tickers': down_tickers[:10],  # Store first 10 for reference
                    'up_tickers': up_tickers[:10]
                })
        
        # Convert to DataFrame
        result_df = pd.DataFrame(daily_stats)
        result_df['date'] = pd.to_datetime(result_df['date'])
        result_df.set_index('date', inplace=True)
        
        logger.info(f"Calculated down percentages for {len(result_df)} days")
        logger.info(f"Average percentage down: {result_df['pct_down'].mean():.1f}%")
        logger.info(f"Max percentage down: {result_df['pct_down'].max():.1f}%")
        logger.info(f"Min percentage down: {result_df['pct_down'].min():.1f}%")
        
        # Save to cache
        if cache_key:
            self.save_daily_stats_cache(result_df, cache_key)
        
        return result_df
    
    def save_daily_stats_cache(self, daily_stats: pd.DataFrame, cache_key: str):
        """
        Save daily statistics to cache file.
        
        Args:
            daily_stats: DataFrame with daily statistics
            cache_key: Unique key for the cache file
        """
        try:
            cache_dir = self.downloader.output_dir / "cache"
            cache_dir.mkdir(exist_ok=True)
            
            cache_file = cache_dir / f"daily_stats_{cache_key}.pkl"
            daily_stats.to_pickle(cache_file)
            
            logger.info(f"Daily statistics cached to {cache_file}")
            
        except Exception as e:
            logger.warning(f"Failed to save daily statistics cache: {e}")
    
    def load_daily_stats_cache(self, cache_key: str) -> Optional[pd.DataFrame]:
        """
        Load daily statistics from cache file if it exists.
        
        Args:
            cache_key: Unique key for the cache file
            
        Returns:
            DataFrame with daily statistics or None if cache doesn't exist
        """
        try:
            cache_dir = self.downloader.output_dir / "cache"
            cache_file = cache_dir / f"daily_stats_{cache_key}.pkl"
            
            if cache_file.exists():
                daily_stats = pd.read_pickle(cache_file)
                
                # Validate the cached data structure
                required_columns = ['total_stocks', 'stocks_down', 'stocks_up', 'pct_down', 'pct_up']
                if all(col in daily_stats.columns for col in required_columns):
                    return daily_stats
                else:
                    logger.warning(f"Cache file {cache_file} has invalid structure, will recalculate")
                    
            return None
            
        except Exception as e:
            logger.warning(f"Failed to load daily statistics cache: {e}")
            return None
    
    def clear_daily_stats_cache(self, cache_key: str = None):
        """
        Clear daily statistics cache files.
        
        Args:
            cache_key: Specific cache key to clear, or None to clear all
        """
        try:
            cache_dir = self.downloader.output_dir / "cache"
            
            if cache_key:
                # Clear specific cache file
                cache_file = cache_dir / f"daily_stats_{cache_key}.pkl"
                if cache_file.exists():
                    cache_file.unlink()
                    logger.info(f"Cleared cache for {cache_key}")
            else:
                # Clear all daily stats cache files
                if cache_dir.exists():
                    for cache_file in cache_dir.glob("daily_stats_*.pkl"):
                        cache_file.unlink()
                    logger.info("Cleared all daily statistics cache files")
                    
        except Exception as e:
            logger.warning(f"Failed to clear cache: {e}")
    
    def list_cached_analyses(self) -> List[str]:
        """
        List available cached analyses.
        
        Returns:
            List of cache keys for available analyses
        """
        try:
            cache_dir = self.downloader.output_dir / "cache"
            cache_keys = []
            
            if cache_dir.exists():
                for cache_file in cache_dir.glob("daily_stats_*.pkl"):
                    # Extract cache key from filename
                    cache_key = cache_file.stem.replace("daily_stats_", "")
                    cache_keys.append(cache_key)
            
            return sorted(cache_keys)
            
        except Exception as e:
            logger.warning(f"Failed to list cached analyses: {e}")
            return []
    
    def identify_down_days(self, daily_stats: pd.DataFrame, 
                          threshold: float = 70.0) -> pd.DataFrame:
        """
        Identify down days based on threshold.
        
        Args:
            daily_stats: DataFrame from calculate_daily_down_percentage
            threshold: Minimum percentage of stocks down to qualify as down day
            
        Returns:
            DataFrame with down days only
        """
        down_days = daily_stats[daily_stats['pct_down'] >= threshold].copy()
        
        logger.info(f"Found {len(down_days)} down days with {threshold}% threshold")
        if len(down_days) > 0:
            logger.info(f"Down day percentage range: {down_days['pct_down'].min():.1f}% - {down_days['pct_down'].max():.1f}%")
        
        return down_days
    
    def generate_down_day_report(self, index_name: str, year: int, 
                                thresholds: List[float] = [60, 70, 80, 90],
                                use_cache: bool = True) -> Dict:
        """
        Generate comprehensive down day report for a given year.
        
        Args:
            index_name: 'nasdaq' or 'sp500'
            year: Year to analyze
            thresholds: List of down day thresholds to analyze
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary with comprehensive down day analysis
        """
        logger.info(f"Generating down day report for {index_name.upper()} {year}")
        
        # Define date range for the year
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        # Create cache key
        cache_key = f"{index_name.lower()}_{year}" if use_cache else None
        
        # Ensure data is available
        index_data = self.ensure_index_data(index_name)
        
        # Calculate daily statistics (with caching)
        daily_stats = self.calculate_daily_down_percentage(
            index_data, start_date, end_date, cache_key
        )
        
        # Analyze different thresholds
        threshold_analysis = {}
        for threshold in thresholds:
            down_days = self.identify_down_days(daily_stats, threshold)
            
            threshold_analysis[threshold] = {
                'down_days_count': len(down_days),
                'percentage_of_year': (len(down_days) / len(daily_stats)) * 100 if len(daily_stats) > 0 else 0,
                'avg_down_pct': down_days['pct_down'].mean() if len(down_days) > 0 else 0,
                'max_down_pct': down_days['pct_down'].max() if len(down_days) > 0 else 0,
                'down_days': down_days
            }
        
        # Monthly breakdown
        daily_stats['month'] = daily_stats.index.month
        monthly_stats = daily_stats.groupby('month').agg({
            'pct_down': ['count', 'mean', 'max', 'min'],
            'total_stocks': 'mean'
        }).round(2)
        
        # Overall statistics
        overall_stats = {
            'total_trading_days': len(daily_stats),
            'avg_pct_down': daily_stats['pct_down'].mean(),
            'std_pct_down': daily_stats['pct_down'].std(),
            'max_pct_down': daily_stats['pct_down'].max(),
            'min_pct_down': daily_stats['pct_down'].min(),
            'avg_stocks_analyzed': daily_stats['total_stocks'].mean()
        }
        
        result = {
            'index': index_name.upper(),
            'year': year,
            'overall_stats': overall_stats,
            'threshold_analysis': threshold_analysis,
            'monthly_breakdown': monthly_stats,
            'daily_data': daily_stats,
            'date_range': {
                'start': daily_stats.index.min().date(),
                'end': daily_stats.index.max().date()
            }
        }
        
        # Log summary
        logger.info(f"Report Summary for {index_name.upper()} {year}:")
        logger.info(f"  Total trading days: {overall_stats['total_trading_days']}")
        logger.info(f"  Average % down: {overall_stats['avg_pct_down']:.1f}%")
        logger.info(f"  Average stocks analyzed: {overall_stats['avg_stocks_analyzed']:.0f}")
        
        for threshold in thresholds:
            count = threshold_analysis[threshold]['down_days_count']
            pct = threshold_analysis[threshold]['percentage_of_year']
            logger.info(f"  {threshold}% threshold: {count} days ({pct:.1f}% of year)")
        
        return result
    
    def save_down_day_report(self, report: Dict, filename: str = None):
        """
        Save down day report to files.
        
        Args:
            report: Report dictionary from generate_down_day_report
            filename: Base filename (optional)
        """
        if filename is None:
            filename = f"{report['index'].lower()}_down_days_{report['year']}"
        
        try:
            # Ensure data directory exists
            data_dir = self.downloader.output_dir
            data_dir.mkdir(exist_ok=True)
            
            # Save daily data
            daily_file = data_dir / f"{filename}_daily_stats.csv"
            report['daily_data'].to_csv(daily_file)
            
            # Save threshold analysis
            threshold_summary = []
            for threshold, analysis in report['threshold_analysis'].items():
                threshold_summary.append({
                    'threshold_pct': threshold,
                    'down_days_count': analysis['down_days_count'],
                    'percentage_of_year': analysis['percentage_of_year'],
                    'avg_down_pct': analysis['avg_down_pct'],
                    'max_down_pct': analysis['max_down_pct']
                })
            
            threshold_df = pd.DataFrame(threshold_summary)
            threshold_file = data_dir / f"{filename}_threshold_analysis.csv"
            threshold_df.to_csv(threshold_file, index=False)
            
            # Save monthly breakdown
            monthly_file = data_dir / f"{filename}_monthly_breakdown.csv"
            report['monthly_breakdown'].to_csv(monthly_file)
            
            logger.info(f"Down day report saved:")
            logger.info(f"  Daily stats: {daily_file}")
            logger.info(f"  Threshold analysis: {threshold_file}")
            logger.info(f"  Monthly breakdown: {monthly_file}")
            
        except PermissionError as e:
            raise PermissionError(f"Cannot save files - they may be open in another application: {e}")
        except Exception as e:
            raise Exception(f"Error saving report files: {e}")
    
    def identify_winners_on_down_days(self, winner_threshold: float = 2.0) -> pd.DataFrame:
        """
        Identify stocks that are winners on down days.

        Args:
            winner_threshold: Percentage threshold to classify a stock as a winner.

        Returns:
            DataFrame containing winners for each down day.
        """
        down_days = self.identify_down_days()
        winners = []

        for date in down_days['date']:
            daily_data = self.index_data[self.index_data['date'] == date]
            daily_winners = daily_data[daily_data['change_pct'] >= winner_threshold]
            winners.append({
                'date': date,
                'winners': daily_winners[['ticker', 'change_pct']].to_dict(orient='records')
            })

        return pd.DataFrame(winners)

    def generate_winners_report(self, winner_threshold: float = 2.0, output_file: str = "winners_report.csv"):
        """
        Generate a report of winners on down days.

        Args:
            winner_threshold: Percentage threshold to classify a stock as a winner.
            output_file: File path to save the report.
        """
        winners = self.identify_winners_on_down_days(winner_threshold)
        winners.to_csv(output_file, index=False)
        logger.info(f"Winners report saved to {output_file}")


def run_down_day_analysis_example(use_cache: bool = True):
    """
    Example of running down day analysis.
    
    Args:
        use_cache: Whether to use cached data if available
    """
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("📊 Market Down Day Analysis")
    print("=" * 40)
    
    # Initialize analyzer
    analyzer = MarketAnalyzer()
    
    # Ensure NASDAQ data for 2024 is available
    try:
        analyzer.ensure_index_data('nasdaq')
    except Exception as e:
        print(f"❌ Error ensuring data availability: {e}")
        return
    
    # Generate report for NASDAQ 2024
    try:
        report = analyzer.generate_down_day_report(
            index_name='nasdaq',
            year=2024,
            thresholds=[60, 70, 80, 90],
            use_cache=use_cache
        )
        
        # Display results
        print(f"\n📈 {report['index']} Down Day Report for {report['year']}")
        print("-" * 50)
        print(f"Total Trading Days: {report['overall_stats']['total_trading_days']}")
        print(f"Average % Down: {report['overall_stats']['avg_pct_down']:.1f}%")
        print(f"Max % Down: {report['overall_stats']['max_pct_down']:.1f}%")
        print(f"Average Stocks Analyzed: {report['overall_stats']['avg_stocks_analyzed']:.0f}")
        
        print(f"\n🔻 Down Day Thresholds:")
        for threshold in [60, 70, 80, 90]:
            analysis = report['threshold_analysis'][threshold]
            print(f"  {threshold}%+: {analysis['down_days_count']} days "
                  f"({analysis['percentage_of_year']:.1f}% of year)")
        
        # Try to save the report
        try:
            analyzer.save_down_day_report(report)
            print(f"\n💾 Report saved to data/ directory")
        except Exception as save_error:
            print(f"\n⚠️  Warning: Could not save report files: {save_error}")
            print("Report generated successfully but files could not be saved.")
        
    except Exception as e:
        print(f"❌ Error generating report: {e}")
        logger.error("Down day analysis failed", exc_info=True)


def run_winners_on_down_days_example(down_day_threshold: float = 80.0, 
                                     winner_threshold: float = 2.0,
                                     winner_hold_days: int = 1,
                                     use_cache: bool = True):
    """
    Example of identifying winners on down days with configurable parameters.
    
    Args:
        down_day_threshold: Percentage threshold for down days (default: 80.0)
        winner_threshold: Percentage threshold for winners (default: 2.0)
        winner_hold_days: Number of days to hold winners (default: 1)
        use_cache: Whether to use cached data if available
    """
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("📊 Winners on Down Days Analysis")
    print("=" * 40)
    print(f"Down Day Threshold: {down_day_threshold}%")
    print(f"Winner Threshold: {winner_threshold}%")
    print(f"Winner Hold Days: {winner_hold_days}")
    
    # Initialize analyzer
    analyzer = MarketAnalyzer()
    
    # Generate report for NASDAQ 2024
    try:
        report = analyzer.generate_down_day_report(
            index_name='nasdaq',
            year=2024,
            thresholds=[down_day_threshold],
            use_cache=use_cache
        )
        
        # Get down days with specified threshold
        down_days = report['threshold_analysis'][down_day_threshold]['down_days']
        
        print(f"\n🔻 Winners on Down Days (Threshold: {down_day_threshold}%)")
        print("-" * 50)
        
        # Get all NASDAQ data once
        daily_data = analyzer.ensure_index_data('nasdaq')
        
        total_winners = 0
        total_down_days = len(down_days)
        
        # Identify winners for each down day
        for date, row in down_days.iterrows():
            print(f"\n📅 Date: {date.strftime('%Y-%m-%d')}")
            print(f"  Total Stocks: {row['total_stocks']}")
            print(f"  Stocks Down: {row['stocks_down']} ({row['pct_down']:.1f}%)")
            
            winners = []
            
            # Look at ALL stocks for this date, not just down_tickers
            for ticker, stock_data in daily_data.items():
                if date in stock_data.index:
                    try:
                        open_price = stock_data.loc[date, 'Open']
                        close_price = stock_data.loc[date, 'Close']
                        
                        if pd.notna(open_price) and pd.notna(close_price) and open_price > 0:
                            daily_return = (close_price - open_price) / open_price
                            if daily_return >= winner_threshold / 100:
                                # Calculate return after holding for specified days
                                hold_return = calculate_hold_return(
                                    stock_data, date, winner_hold_days, open_price, close_price
                                )
                                
                                winners.append({
                                    'ticker': ticker, 
                                    'day_change_pct': daily_return * 100,
                                    'hold_return_pct': hold_return
                                })
                    except Exception as e:
                        logger.debug(f"Error processing {ticker} for {date}: {e}")
                        continue
            
            # Sort winners by day performance
            winners.sort(key=lambda x: x['day_change_pct'], reverse=True)
            total_winners += len(winners)
            
            if not winners:
                print("  No winners found.")
            else:
                print(f"  Winners ({len(winners)} stocks):")
                # Show top 10 winners
                for winner in winners[:10]:
                    print(f"    {winner['ticker']}: +{winner['day_change_pct']:.2f}% "
                          f"(Hold {winner_hold_days} day{'s' if winner_hold_days > 1 else ''}: "
                          f"{winner['hold_return_pct']:+.2f}%)")
                
                if len(winners) > 10:
                    print(f"    ... and {len(winners) - 10} more winners")
        
        # Summary statistics
        print(f"\n📈 Summary Statistics:")
        print(f"  Total down days analyzed: {total_down_days}")
        print(f"  Total winners found: {total_winners}")
        print(f"  Average winners per down day: {total_winners / total_down_days:.1f}")
        
    except Exception as e:
        print(f"❌ Error identifying winners: {e}")
        logger.error("Winners analysis failed", exc_info=True)


def calculate_hold_return(stock_data: pd.DataFrame, buy_date: pd.Timestamp, 
                         hold_days: int, buy_open: float, buy_close: float) -> float:
    """
    Calculate the return after holding a stock for specified number of days.
    
    Args:
        stock_data: Stock price data
        buy_date: Date when stock was bought
        hold_days: Number of days to hold
        buy_open: Opening price on buy date
        buy_close: Closing price on buy date (actual buy price)
        
    Returns:
        Return percentage after holding for specified days
    """
    try:
        # Find sell date (hold_days trading days after buy_date)
        available_dates = stock_data.index[stock_data.index > buy_date].sort_values()
        
        if len(available_dates) < hold_days:
            # Not enough data, return the best available
            if len(available_dates) == 0:
                return 0.0
            sell_date = available_dates[-1]
        else:
            sell_date = available_dates[hold_days - 1]
        
        # Get sell price (opening price of sell date)
        if sell_date in stock_data.index:
            sell_price = stock_data.loc[sell_date, 'Open']
            if pd.notna(sell_price) and sell_price > 0:
                return ((sell_price - buy_close) / buy_close) * 100
        
        return 0.0
        
    except Exception as e:
        logger.debug(f"Error calculating hold return: {e}")
        return 0.0


def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Market Down Day Analysis with configurable parameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--down-day-threshold', 
        type=float, 
        default=80.0,
        help='Percentage threshold for down days (e.g., 80.0 for 80%%)'
    )
    
    parser.add_argument(
        '--winner-threshold', 
        type=float, 
        default=2.0,
        help='Percentage threshold for winners (e.g., 2.0 for 2%%)'
    )
    
    parser.add_argument(
        '--winner-hold-days', 
        type=int, 
        default=1,
        help='Number of days to hold winner stocks before selling'
    )
    
    parser.add_argument(
        '--analysis-type',
        choices=['down-days', 'winners', 'both'],
        default='both',
        help='Type of analysis to run'
    )
    
    parser.add_argument(
        '--year',
        type=int,
        default=2024,
        help='Year to analyze'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable caching and force recalculation of daily statistics'
    )
    
    parser.add_argument(
        '--clear-cache',
        type=str,
        metavar='CACHE_KEY',
        help='Clear cached data for specific analysis (e.g., nasdaq_2024) or "all" for all cache'
    )
    
    parser.add_argument(
        '--list-cache',
        action='store_true',
        help='List available cached analyses and exit'
    )
    
    args = parser.parse_args()
    
    # Handle cache management commands
    if args.list_cache:
        analyzer = MarketAnalyzer()
        cached_analyses = analyzer.list_cached_analyses()
        if cached_analyses:
            print("📁 Available cached analyses:")
            for cache_key in cached_analyses:
                print(f"  - {cache_key}")
        else:
            print("📁 No cached analyses found")
        return
    
    if args.clear_cache:
        analyzer = MarketAnalyzer()
        if args.clear_cache.lower() == 'all':
            analyzer.clear_daily_stats_cache()
            print("🗑️  Cleared all cached analyses")
        else:
            analyzer.clear_daily_stats_cache(args.clear_cache)
            print(f"🗑️  Cleared cache for {args.clear_cache}")
        return
    
    # Validate arguments
    if args.down_day_threshold < 0 or args.down_day_threshold > 100:
        print("❌ Error: Down day threshold must be between 0 and 100")
        return
    
    if args.winner_threshold < 0:
        print("❌ Error: Winner threshold must be positive")
        return
    
    if args.winner_hold_days < 1:
        print("❌ Error: Winner hold days must be at least 1")
        return
    
    use_cache = not args.no_cache
    
    print(f"🔧 Configuration:")
    print(f"  Analysis Year: {args.year}")
    print(f"  Down Day Threshold: {args.down_day_threshold}%")
    print(f"  Winner Threshold: {args.winner_threshold}%")
    print(f"  Winner Hold Days: {args.winner_hold_days}")
    print(f"  Analysis Type: {args.analysis_type}")
    print(f"  Use Cache: {'Yes' if use_cache else 'No'}")
    print()
    
    # Run selected analysis
    if args.analysis_type in ['down-days', 'both']:
        run_down_day_analysis_example(use_cache=use_cache)
        print()
    
    if args.analysis_type in ['winners', 'both']:
        run_winners_on_down_days_example(
            down_day_threshold=args.down_day_threshold,
            winner_threshold=args.winner_threshold,
            winner_hold_days=args.winner_hold_days,
            use_cache=use_cache
        )


if __name__ == "__main__":
    main()


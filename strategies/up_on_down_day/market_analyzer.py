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
from pathlib import Path
from datetime import datetime, date
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
        
        Args:
            index_name: 'nasdaq' or 'sp500'
            
        Returns:
            Dictionary of {ticker: DataFrame} for all index stocks
        """
        logger.info(f"Ensuring data availability for {index_name.upper()} index")
        
        if index_name.lower() == 'nasdaq':
            tickers = self.downloader.get_nasdaq_tickers()
            subfolder = 'nasdaq'
        elif index_name.lower() == 'sp500':
            tickers = self.downloader.get_sp500_tickers()
            subfolder = 'sp500'
        else:
            raise ValueError(f"Unsupported index: {index_name}")
        
        logger.info(f"Found {len(tickers)} tickers for {index_name.upper()}")
        
        # Check which data we already have
        available_data = {}
        missing_tickers = []
        
        for ticker in tickers:
            data_file = self.downloader.output_dir / subfolder / f"{ticker}_historical_data.csv"
            if data_file.exists():
                try:
                    data = pd.read_csv(data_file)
                    data['Date'] = pd.to_datetime(data['Date'])
                    data.set_index('Date', inplace=True)
                    available_data[ticker] = data
                except Exception as e:
                    logger.warning(f"Error loading {ticker}: {e}")
                    missing_tickers.append(ticker)
            else:
                missing_tickers.append(ticker)
        
        logger.info(f"Found existing data for {len(available_data)} stocks")
        
        # Download missing data
        if missing_tickers:
            logger.info(f"Downloading missing data for {len(missing_tickers)} stocks")
            
            for i, ticker in enumerate(missing_tickers, 1):
                try:
                    logger.info(f"Downloading {ticker} ({i}/{len(missing_tickers)})")
                    data = self.downloader.download_stock_data_incremental(ticker, subfolder)
                    if data is not None:
                        data['Date'] = pd.to_datetime(data['Date'])
                        data.set_index('Date', inplace=True)
                        available_data[ticker] = data
                except Exception as e:
                    logger.error(f"Failed to download {ticker}: {e}")
                    continue
        
        logger.info(f"Total available data: {len(available_data)} stocks")
        return available_data
    
    def calculate_daily_down_percentage(self, index_data: Dict[str, pd.DataFrame], 
                                      start_date: Union[str, date] = None,
                                      end_date: Union[str, date] = None) -> pd.DataFrame:
        """
        Calculate the percentage of stocks that are down each day.
        
        Args:
            index_data: Dictionary of {ticker: DataFrame} with stock data
            start_date: Start date for analysis (optional)
            end_date: End date for analysis (optional)
            
        Returns:
            DataFrame with daily down percentages
        """
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
        
        return result_df
    
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
                                thresholds: List[float] = [60, 70, 80, 90]) -> Dict:
        """
        Generate comprehensive down day report for a given year.
        
        Args:
            index_name: 'nasdaq' or 'sp500'
            year: Year to analyze
            thresholds: List of down day thresholds to analyze
            
        Returns:
            Dictionary with comprehensive down day analysis
        """
        logger.info(f"Generating down day report for {index_name.upper()} {year}")
        
        # Define date range for the year
        start_date = f"{year}-01-01"
        end_date = f"{year}-12-31"
        
        # Ensure data is available
        index_data = self.ensure_index_data(index_name)
        
        # Calculate daily statistics
        daily_stats = self.calculate_daily_down_percentage(index_data, start_date, end_date)
        
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


def run_down_day_analysis_example():
    """Example of running down day analysis."""
    
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
            thresholds=[60, 70, 80, 90]
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


def run_winners_on_down_days_example():
    """Example of identifying winners on down days."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("📊 Winners on Down Days Analysis")
    print("=" * 40)
    
    # Initialize analyzer
    analyzer = MarketAnalyzer()
    
    # Generate report for NASDAQ 2024
    try:
        report = analyzer.generate_down_day_report(
            index_name='nasdaq',
            year=2024,
            thresholds=[80]  # Down day threshold of 80%
        )
        
        # Get down days with 80% threshold
        down_days = report['threshold_analysis'][80]['down_days']
        
        print(f"\n🔻 Winners on Down Days (Threshold: 80%)")
        print("-" * 50)
        
        # Get all NASDAQ data once
        daily_data = analyzer.ensure_index_data('nasdaq')
        
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
                            # Using a winner threshold of 2.0% 
                            if daily_return >= 0.02:
                                winners.append({
                                    'ticker': ticker, 
                                    'change_pct': daily_return * 100
                                })
                    except Exception as e:
                        logger.debug(f"Error processing {ticker} for {date}: {e}")
                        continue
            
            # Sort winners by performance
            winners.sort(key=lambda x: x['change_pct'], reverse=True)
            
            if not winners:
                print("  No winners found.")
            else:
                print(f"  Winners ({len(winners)} stocks):")
                # Show top 10 winners
                for winner in winners[:10]:
                    print(f"    {winner['ticker']}: +{winner['change_pct']:.2f}%")
                
                if len(winners) > 10:
                    print(f"    ... and {len(winners) - 10} more winners")
        
    except Exception as e:
        print(f"❌ Error identifying winners: {e}")
        logger.error("Winners analysis failed", exc_info=True)


if __name__ == "__main__":
    run_down_day_analysis_example()
    run_winners_on_down_days_example()


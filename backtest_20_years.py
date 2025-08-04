"""
20-Year Backtesting Analysis for Close-to-Open Strategy

This script tests the strategy over a 20-year period to capture:
- Multiple market cycles (bull/bear markets)
- Different economic conditions
- Various volatility regimes
- Long-term performance characteristics
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def backtest_20_years(symbol, initial_capital=10000):
    """
    Comprehensive 20-year backtest of the close-to-open strategy.
    
    Args:
        symbol: Stock ticker (SPY, QQQ, etc.)
        initial_capital: Starting capital
    
    Returns:
        Detailed results dictionary
    """
    try:
        print(f"\n🔍 Fetching 20 years of data for {symbol}...")
        
        # Fetch 20 years of data (max period)
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="20y", interval="1d")
        
        if data.empty:
            return {"error": f"No data available for {symbol}"}
        
        print(f"✅ Data fetched: {len(data)} trading days from {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        
        # Prepare data for strategy
        data['Next_Open'] = data['Open'].shift(-1)
        data = data[:-1]  # Remove last row without next open
        
        # Strategy execution
        capital = initial_capital
        shares = 0
        trades = []
        daily_portfolio_values = []
        yearly_summary = []
        
        current_year = None
        year_start_value = initial_capital
        year_trades = 0
        
        total_trading_days = len(data)
        print(f"📊 Executing strategy over {total_trading_days} trading days...")
        
        for i, (date, row) in enumerate(data.iterrows()):
            close_price = row['Close']
            next_open = row['Next_Open']
            
            # Track yearly performance
            if current_year != date.year:
                if current_year is not None:
                    # Save previous year's summary
                    year_end_value = daily_portfolio_values[-1]['portfolio_value'] if daily_portfolio_values else year_start_value
                    year_return = ((year_end_value - year_start_value) / year_start_value) * 100
                    
                    # Buy and hold for the year
                    year_data = data[data.index.year == current_year]
                    if len(year_data) > 0:
                        year_bh_return = ((year_data['Close'].iloc[-1] - year_data['Close'].iloc[0]) / year_data['Close'].iloc[0]) * 100
                    else:
                        year_bh_return = 0
                    
                    yearly_summary.append({
                        'year': current_year,
                        'start_value': year_start_value,
                        'end_value': year_end_value,
                        'strategy_return': year_return,
                        'buy_hold_return': year_bh_return,
                        'excess_return': year_return - year_bh_return,
                        'trades': year_trades
                    })
                
                # Start new year
                current_year = date.year
                year_start_value = capital + (shares * close_price)
                year_trades = 0
                print(f"📅 Processing year {current_year}...")
            
            if pd.notna(next_open):
                # Buy at close
                if capital >= close_price:
                    shares_to_buy = int(capital / close_price)
                    if shares_to_buy > 0:
                        cost = shares_to_buy * close_price
                        capital -= cost
                        shares += shares_to_buy
                        year_trades += 1
                        
                        trades.append({
                            'date': date,
                            'action': 'BUY',
                            'price': close_price,
                            'shares': shares_to_buy,
                            'value': cost
                        })
                
                # Sell at next open
                if shares > 0:
                    proceeds = shares * next_open
                    capital += proceeds
                    year_trades += 1
                    
                    trades.append({
                        'date': date + timedelta(days=1),
                        'action': 'SELL',
                        'price': next_open,
                        'shares': shares,
                        'value': proceeds
                    })
                    
                    shares = 0
                
                # Record daily portfolio value
                portfolio_value = capital + (shares * close_price)
                daily_portfolio_values.append({
                    'date': date,
                    'portfolio_value': portfolio_value,
                    'capital': capital,
                    'stock_price': close_price,
                    'year': date.year
                })
        
        # Final year summary
        if daily_portfolio_values:
            final_value = daily_portfolio_values[-1]['portfolio_value']
            year_return = ((final_value - year_start_value) / year_start_value) * 100
            
            year_data = data[data.index.year == current_year]
            if len(year_data) > 0:
                year_bh_return = ((year_data['Close'].iloc[-1] - year_data['Close'].iloc[0]) / year_data['Close'].iloc[0]) * 100
            else:
                year_bh_return = 0
            
            yearly_summary.append({
                'year': current_year,
                'start_value': year_start_value,
                'end_value': final_value,
                'strategy_return': year_return,
                'buy_hold_return': year_bh_return,
                'excess_return': year_return - year_bh_return,
                'trades': year_trades
            })
        
        # Calculate overall metrics
        final_value = daily_portfolio_values[-1]['portfolio_value'] if daily_portfolio_values else initial_capital
        total_return = final_value - initial_capital
        total_return_pct = (total_return / initial_capital) * 100
        
        # Annualized return
        years = len(data) / 252  # Approximate trading days per year
        annualized_return = ((final_value / initial_capital) ** (1/years) - 1) * 100
        
        # Overall buy and hold
        initial_price = data['Close'].iloc[0]
        final_price = data['Close'].iloc[-1]
        buy_hold_total = ((final_price - initial_price) / initial_price) * 100
        buy_hold_annualized = ((final_price / initial_price) ** (1/years) - 1) * 100
        
        # Risk metrics
        values_df = pd.DataFrame(daily_portfolio_values)
        values_df['daily_return'] = values_df['portfolio_value'].pct_change()
        daily_returns = values_df['daily_return'].dropna()
        
        volatility = daily_returns.std() * np.sqrt(252) * 100  # Annualized volatility
        sharpe_ratio = (annualized_return / 100) / (volatility / 100) if volatility > 0 else 0
        
        # Maximum drawdown
        values_df['cummax'] = values_df['portfolio_value'].cummax()
        values_df['drawdown'] = (values_df['portfolio_value'] - values_df['cummax']) / values_df['cummax']
        max_drawdown = values_df['drawdown'].min() * 100
        
        # Win rate
        profitable_pairs = sum(1 for i in range(0, len(trades)-1, 2) 
                              if i+1 < len(trades) and trades[i+1]['value'] > trades[i]['value'])
        total_pairs = len(trades) // 2
        win_rate = (profitable_pairs / total_pairs * 100) if total_pairs > 0 else 0
        
        print(f"✅ Analysis complete!")
        
        return {
            'symbol': symbol,
            'period': f"{data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}",
            'trading_days': len(data),
            'years': years,
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'annualized_return': annualized_return,
            'buy_hold_total': buy_hold_total,
            'buy_hold_annualized': buy_hold_annualized,
            'excess_return_total': total_return_pct - buy_hold_total,
            'excess_return_annualized': annualized_return - buy_hold_annualized,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'total_pairs': total_pairs,
            'yearly_summary': yearly_summary,
            'daily_values': daily_portfolio_values
        }
        
    except Exception as e:
        return {"error": str(e)}

def analyze_performance_by_decade(yearly_data):
    """Analyze performance by decade."""
    decades = {}
    
    for year_data in yearly_data:
        decade = (year_data['year'] // 10) * 10
        if decade not in decades:
            decades[decade] = []
        decades[decade].append(year_data)
    
    decade_summary = []
    for decade, years in decades.items():
        if years:
            avg_strategy = np.mean([y['strategy_return'] for y in years])
            avg_buy_hold = np.mean([y['buy_hold_return'] for y in years])
            total_trades = sum([y['trades'] for y in years])
            
            decade_summary.append({
                'decade': f"{decade}s",
                'years_count': len(years),
                'avg_strategy_return': avg_strategy,
                'avg_buy_hold_return': avg_buy_hold,
                'avg_excess_return': avg_strategy - avg_buy_hold,
                'total_trades': total_trades
            })
    
    return decade_summary

def main():
    """Run 20-year backtesting analysis."""
    print("=" * 70)
    print("20-YEAR CLOSE-TO-OPEN STRATEGY ANALYSIS")
    print("=" * 70)
    print("Strategy: Buy at close, sell at next open (daily)")
    print("Initial Capital: $10,000")
    print("Testing Period: Maximum available (up to 20 years)")
    print()
    
    symbols = ['SPY', 'QQQ']
    
    for symbol in symbols:
        print(f"\n🎯 ANALYZING {symbol} - 20 YEAR BACKTEST")
        print("=" * 50)
        
        result = backtest_20_years(symbol)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            continue
        
        # Display overall results
        print(f"\n📈 OVERALL PERFORMANCE")
        print("-" * 30)
        print(f"Period: {result['period']}")
        print(f"Trading Days: {result['trading_days']:,}")
        print(f"Years: {result['years']:.1f}")
        print()
        print(f"Initial Capital: ${result['initial_capital']:,}")
        print(f"Final Value: ${result['final_value']:,.2f}")
        print(f"Total Return: ${result['total_return']:,.2f} ({result['total_return_pct']:+.2f}%)")
        print(f"Annualized Return: {result['annualized_return']:+.2f}%")
        print()
        print(f"Buy & Hold Total: {result['buy_hold_total']:+.2f}%")
        print(f"Buy & Hold Annualized: {result['buy_hold_annualized']:+.2f}%")
        print(f"Excess Return (Total): {result['excess_return_total']:+.2f}%")
        print(f"Excess Return (Annualized): {result['excess_return_annualized']:+.2f}%")
        print()
        print(f"Risk Metrics:")
        print(f"  Volatility: {result['volatility']:.1f}%")
        print(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {result['max_drawdown']:.1f}%")
        print(f"  Win Rate: {result['win_rate']:.1f}%")
        print(f"  Total Trades: {result['total_trades']:,}")
        
        # Performance indicator
        if result['excess_return_annualized'] > 0:
            print(f"\n🎉 Strategy OUTPERFORMED buy & hold by {result['excess_return_annualized']:+.2f}% annually!")
        else:
            print(f"\n📉 Strategy underperformed buy & hold by {abs(result['excess_return_annualized']):.2f}% annually")
        
        # Yearly breakdown
        print(f"\n📅 YEAR-BY-YEAR PERFORMANCE")
        print("-" * 50)
        yearly_df = pd.DataFrame(result['yearly_summary'])
        if not yearly_df.empty:
            # Show recent years and summary stats
            print("Recent Years Performance:")
            recent_years = yearly_df.tail(10)
            for _, year in recent_years.iterrows():
                print(f"{year['year']}: Strategy {year['strategy_return']:+6.2f}% | Buy&Hold {year['buy_hold_return']:+6.2f}% | Excess {year['excess_return']:+6.2f}%")
            
            # Decade analysis
            decade_summary = analyze_performance_by_decade(result['yearly_summary'])
            if decade_summary:
                print(f"\n📊 PERFORMANCE BY DECADE")
                print("-" * 30)
                for decade in decade_summary:
                    print(f"{decade['decade']}: Avg Strategy {decade['avg_strategy_return']:+.2f}% | Avg B&H {decade['avg_buy_hold_return']:+.2f}% | Excess {decade['avg_excess_return']:+.2f}%")
        
        # Best and worst years
        if not yearly_df.empty:
            best_year = yearly_df.loc[yearly_df['strategy_return'].idxmax()]
            worst_year = yearly_df.loc[yearly_df['strategy_return'].idxmin()]
            
            print(f"\n🏆 BEST YEAR: {best_year['year']} ({best_year['strategy_return']:+.2f}%)")
            print(f"💥 WORST YEAR: {worst_year['year']} ({worst_year['strategy_return']:+.2f}%)")
        
        print(f"\n" + "=" * 50)

if __name__ == "__main__":
    main()

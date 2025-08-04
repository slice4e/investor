"""
20-Year Open-to-Close Strategy Analysis

Compare the long-term performance of:
1. Open-to-Close (intraday momentum)
2. Close-to-Open (overnight gaps)
3. Buy and Hold
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def backtest_open_to_close_20y(symbol, initial_capital=10000):
    """
    20-year backtest of open-to-close strategy.
    """
    try:
        print(f"\n🔍 Fetching 20 years of data for {symbol} (Open-to-Close)...")
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="20y", interval="1d")
        
        if data.empty:
            return {"error": f"No data available for {symbol}"}
        
        print(f"✅ Data fetched: {len(data)} trading days from {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        
        # Strategy execution
        capital = initial_capital
        trades = []
        daily_portfolio_values = []
        yearly_summary = []
        
        current_year = None
        year_start_value = initial_capital
        year_trades = 0
        winning_days = 0
        losing_days = 0
        
        for date, row in data.iterrows():
            open_price = row['Open']
            close_price = row['Close']
            
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
                year_start_value = capital
                year_trades = 0
                print(f"📅 Processing year {current_year}...")
            
            # Skip if missing data
            if pd.isna(open_price) or pd.isna(close_price):
                continue
            
            # Execute open-to-close trade
            if capital >= open_price:
                shares_bought = int(capital / open_price)
                if shares_bought > 0:
                    buy_cost = shares_bought * open_price
                    sell_proceeds = shares_bought * close_price
                    
                    capital = capital - buy_cost + sell_proceeds
                    year_trades += 2  # Buy and sell
                    
                    # Track win/loss
                    if close_price > open_price:
                        winning_days += 1
                    elif close_price < open_price:
                        losing_days += 1
                    
                    trades.extend([
                        {
                            'date': date,
                            'action': 'BUY',
                            'price': open_price,
                            'shares': shares_bought,
                            'value': buy_cost
                        },
                        {
                            'date': date,
                            'action': 'SELL',
                            'price': close_price,
                            'shares': shares_bought,
                            'value': sell_proceeds
                        }
                    ])
            
            # Record daily portfolio value
            daily_portfolio_values.append({
                'date': date,
                'portfolio_value': capital,
                'open_price': open_price,
                'close_price': close_price,
                'daily_return': (close_price - open_price) / open_price if open_price > 0 else 0,
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
        years = len(data) / 252
        annualized_return = ((final_value / initial_capital) ** (1/years) - 1) * 100
        
        # Buy and hold comparison
        initial_price = data['Close'].iloc[0]
        final_price = data['Close'].iloc[-1]
        buy_hold_total = ((final_price - initial_price) / initial_price) * 100
        buy_hold_annualized = ((final_price / initial_price) ** (1/years) - 1) * 100
        
        # Risk metrics
        values_df = pd.DataFrame(daily_portfolio_values)
        values_df['daily_return'] = values_df['portfolio_value'].pct_change()
        daily_returns = values_df['daily_return'].dropna()
        
        volatility = daily_returns.std() * np.sqrt(252) * 100
        sharpe_ratio = (annualized_return / 100) / (volatility / 100) if volatility > 0 else 0
        
        # Maximum drawdown
        values_df['cummax'] = values_df['portfolio_value'].cummax()
        values_df['drawdown'] = (values_df['portfolio_value'] - values_df['cummax']) / values_df['cummax']
        max_drawdown = values_df['drawdown'].min() * 100
        
        # Win rate
        total_trading_days = winning_days + losing_days
        win_rate = (winning_days / total_trading_days * 100) if total_trading_days > 0 else 0
        
        print(f"✅ Open-to-Close analysis complete!")
        
        return {
            'symbol': symbol,
            'strategy': 'Open-to-Close',
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
            'winning_days': winning_days,
            'losing_days': losing_days,
            'total_trades': len(trades),
            'yearly_summary': yearly_summary,
            'daily_values': daily_portfolio_values
        }
        
    except Exception as e:
        return {"error": str(e)}

def backtest_close_to_open_20y(symbol, initial_capital=10000):
    """
    20-year backtest of close-to-open strategy for comparison.
    """
    try:
        print(f"🔍 Fetching 20 years of data for {symbol} (Close-to-Open)...")
        
        ticker = yf.Ticker(symbol)
        data = ticker.history(period="20y", interval="1d")
        data['Next_Open'] = data['Open'].shift(-1)
        data = data[:-1]  # Remove last row
        
        capital = initial_capital
        shares = 0
        
        for _, row in data.iterrows():
            close_price = row['Close']
            next_open = row['Next_Open']
            
            if pd.notna(next_open):
                # Buy at close
                if capital >= close_price:
                    shares_to_buy = int(capital / close_price)
                    if shares_to_buy > 0:
                        cost = shares_to_buy * close_price
                        capital -= cost
                        shares += shares_to_buy
                
                # Sell at next open
                if shares > 0:
                    proceeds = shares * next_open
                    capital += proceeds
                    shares = 0
        
        final_value = capital + (shares * data['Close'].iloc[-1])
        total_return_pct = ((final_value - initial_capital) / initial_capital) * 100
        years = len(data) / 252
        annualized_return = ((final_value / initial_capital) ** (1/years) - 1) * 100
        
        return {
            'symbol': symbol,
            'strategy': 'Close-to-Open',
            'final_value': final_value,
            'total_return_pct': total_return_pct,
            'annualized_return': annualized_return
        }
        
    except Exception as e:
        return {"error": str(e)}

def compare_20_year_strategies():
    """
    Compare both strategies over 20 years.
    """
    print("=" * 80)
    print("20-YEAR STRATEGY COMPARISON: OPEN-TO-CLOSE vs CLOSE-TO-OPEN")
    print("=" * 80)
    
    symbols = ['SPY', 'QQQ']
    
    for symbol in symbols:
        print(f"\n🎯 {symbol} - 20 YEAR COMPREHENSIVE ANALYSIS")
        print("=" * 60)
        
        # Open-to-Close strategy
        otc_results = backtest_open_to_close_20y(symbol)
        
        # Close-to-Open strategy  
        cto_results = backtest_close_to_open_20y(symbol)
        
        if 'error' in otc_results or 'error' in cto_results:
            print("❌ Error in analysis")
            continue
        
        # Display comparison
        print(f"\n📊 STRATEGY PERFORMANCE COMPARISON")
        print("-" * 50)
        print(f"Period: {otc_results['period']}")
        print(f"Trading Days: {otc_results['trading_days']:,}")
        print(f"Years: {otc_results['years']:.1f}")
        print()
        
        # Results table
        print(f"{'Strategy':<20} {'Final Value':<15} {'Total Return':<15} {'Annualized':<12}")
        print("-" * 65)
        print(f"{'Open-to-Close':<20} ${otc_results['final_value']:>12,.2f} {otc_results['total_return_pct']:>+8.2f}%    {otc_results['annualized_return']:>+8.2f}%")
        print(f"{'Close-to-Open':<20} ${cto_results['final_value']:>12,.2f} {cto_results['total_return_pct']:>+8.2f}%    {cto_results['annualized_return']:>+8.2f}%")
        print(f"{'Buy & Hold':<20} ${(10000 * (1 + otc_results['buy_hold_total']/100)):>12,.2f} {otc_results['buy_hold_total']:>+8.2f}%    {otc_results['buy_hold_annualized']:>+8.2f}%")
        print()
        
        # Risk and performance metrics
        print(f"📈 DETAILED METRICS - OPEN-TO-CLOSE")
        print("-" * 40)
        print(f"Volatility: {otc_results['volatility']:.1f}%")
        print(f"Sharpe Ratio: {otc_results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {otc_results['max_drawdown']:.1f}%")
        print(f"Win Rate: {otc_results['win_rate']:.1f}%")
        print(f"Total Trades: {otc_results['total_trades']:,}")
        print()
        
        # Strategy rankings
        strategies = [
            ('Open-to-Close', otc_results['annualized_return']),
            ('Close-to-Open', cto_results['annualized_return']),
            ('Buy & Hold', otc_results['buy_hold_annualized'])
        ]
        strategies.sort(key=lambda x: x[1], reverse=True)
        
        print(f"🏆 STRATEGY RANKINGS (by annualized return)")
        print("-" * 45)
        for i, (strategy, return_rate) in enumerate(strategies, 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
            print(f"{emoji} {i}. {strategy}: {return_rate:+.2f}% annually")
        
        # Key insights
        print(f"\n💡 KEY INSIGHTS FOR {symbol}")
        print("-" * 30)
        
        if otc_results['annualized_return'] > cto_results['annualized_return']:
            diff = otc_results['annualized_return'] - cto_results['annualized_return']
            print(f"• Open-to-Close outperformed Close-to-Open by {diff:.2f}% annually")
        else:
            diff = cto_results['annualized_return'] - otc_results['annualized_return']
            print(f"• Close-to-Open outperformed Open-to-Close by {diff:.2f}% annually")
        
        if otc_results['win_rate'] > 50:
            print(f"• Open-to-Close had a positive win rate of {otc_results['win_rate']:.1f}%")
        
        if otc_results['volatility'] < 15:
            print(f"• Open-to-Close showed relatively low volatility ({otc_results['volatility']:.1f}%)")
        elif otc_results['volatility'] > 20:
            print(f"• Open-to-Close showed high volatility ({otc_results['volatility']:.1f}%)")
        
        print()

if __name__ == "__main__":
    compare_20_year_strategies()

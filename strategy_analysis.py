"""
Close-to-Open Strategy Analysis for SPY and QQQ

This script implements and tests your intraday strategy:
- Buy at market close
- Sell at next market open
- Execute daily with all available capital
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def analyze_close_to_open_strategy(symbol, period="1y", initial_capital=10000):
    """
    Analyze the close-to-open strategy for a given symbol.
    
    Args:
        symbol: Stock ticker (e.g., 'SPY', 'QQQ')
        period: Time period for analysis
        initial_capital: Starting capital
    
    Returns:
        Dictionary with results
    """
    try:
        # Fetch data
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        
        if data.empty:
            return {"error": f"No data for {symbol}"}
        
        # Add next day's open
        data['Next_Open'] = data['Open'].shift(-1)
        data = data[:-1]  # Remove last row without next open
        
        # Strategy simulation
        capital = initial_capital
        shares = 0
        trades = []
        daily_values = []
        
        for date, row in data.iterrows():
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
                daily_values.append({
                    'date': date,
                    'value': portfolio_value,
                    'capital': capital,
                    'stock_price': close_price
                })
        
        # Calculate metrics
        final_value = daily_values[-1]['value'] if daily_values else initial_capital
        total_return = final_value - initial_capital
        return_pct = (total_return / initial_capital) * 100
        
        # Buy and hold comparison
        initial_price = data['Close'].iloc[0]
        final_price = data['Close'].iloc[-1]
        buy_hold_return = ((final_price - initial_price) / initial_price) * 100
        
        # Calculate daily returns for volatility
        values_df = pd.DataFrame(daily_values)
        values_df['daily_return'] = values_df['value'].pct_change()
        volatility = values_df['daily_return'].std() * np.sqrt(252) * 100  # Annualized %
        
        # Win rate
        profitable_trades = sum(1 for i in range(0, len(trades)-1, 2) 
                               if i+1 < len(trades) and trades[i+1]['value'] > trades[i]['value'])
        total_trade_pairs = len(trades) // 2
        win_rate = (profitable_trades / total_trade_pairs * 100) if total_trade_pairs > 0 else 0
        
        return {
            'symbol': symbol,
            'period': period,
            'initial_capital': initial_capital,
            'final_value': final_value,
            'total_return': total_return,
            'return_percentage': return_pct,
            'buy_hold_return': buy_hold_return,
            'excess_return': return_pct - buy_hold_return,
            'volatility': volatility,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'trade_pairs': total_trade_pairs,
            'trades': trades[-10:],  # Last 10 trades
            'daily_values': daily_values[-10:]  # Last 10 days
        }
        
    except Exception as e:
        return {"error": str(e)}

def main():
    """Run comprehensive analysis."""
    print("=" * 60)
    print("CLOSE-TO-OPEN STRATEGY ANALYSIS")
    print("=" * 60)
    print("Strategy: Buy at close, sell at next open (daily)")
    print("Initial Capital: $10,000")
    print()
    
    symbols = ['SPY', 'QQQ']
    periods = ['3mo', '6mo', '1y']
    
    results_summary = []
    
    for symbol in symbols:
        print(f"\n📊 ANALYZING {symbol}")
        print("-" * 30)
        
        for period in periods:
            print(f"\n--- {period.upper()} Analysis ---")
            
            result = analyze_close_to_open_strategy(symbol, period)
            
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
                continue
            
            # Display results
            print(f"Period: {result['period']}")
            print(f"Final Value: ${result['final_value']:,.2f}")
            print(f"Total Return: ${result['total_return']:,.2f} ({result['return_percentage']:+.2f}%)")
            print(f"Buy & Hold: {result['buy_hold_return']:+.2f}%")
            print(f"Excess Return: {result['excess_return']:+.2f}%")
            print(f"Volatility: {result['volatility']:.1f}%")
            print(f"Win Rate: {result['win_rate']:.1f}%")
            print(f"Total Trades: {result['total_trades']}")
            
            # Performance indicator
            if result['excess_return'] > 0:
                print("🎉 Strategy OUTPERFORMED buy & hold")
            else:
                print("📉 Strategy underperformed buy & hold")
            
            # Add to summary
            results_summary.append({
                'Symbol': symbol,
                'Period': period,
                'Strategy Return': f"{result['return_percentage']:+.2f}%",
                'Buy & Hold': f"{result['buy_hold_return']:+.2f}%",
                'Excess Return': f"{result['excess_return']:+.2f}%",
                'Win Rate': f"{result['win_rate']:.1f}%",
                'Trades': result['total_trades']
            })
    
    # Summary table
    print(f"\n" + "=" * 60)
    print("STRATEGY PERFORMANCE SUMMARY")
    print("=" * 60)
    
    if results_summary:
        df = pd.DataFrame(results_summary)
        print(df.to_string(index=False))
    
    print(f"\n" + "=" * 60)
    print("STRATEGY NOTES:")
    print("• Excess Return = Strategy Return - Buy & Hold Return")
    print("• Positive excess return means strategy beat buy & hold")
    print("• Win Rate = % of profitable trade pairs")
    print("• Each trade pair = buy at close + sell at next open")
    print("=" * 60)

if __name__ == "__main__":
    main()

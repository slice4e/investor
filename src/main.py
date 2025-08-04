"""
Main entry point for the Stock Investor application.
"""

import click
import logging
import sys
from src.data_fetcher import StockDataFetcher
from src.portfolio import Portfolio
from src.analyzer import StockAnalyzer
from src.backtester import Backtester, CloseToOpenStrategy


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('stock_investor.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )


@click.group()
def cli():
    """Stock Investor CLI Application."""
    setup_logging()


@cli.command()
@click.argument('symbol')
@click.option('--period', default='1y', help='Data period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y)')
def analyze(symbol, period):
    """Analyze a stock symbol."""
    analyzer = StockAnalyzer()
    click.echo(f"Analyzing {symbol} for period {period}...")
    
    analysis = analyzer.analyze_stock(symbol, period)
    if not analysis:
        click.echo(f"No data found for {symbol}")
        return
    
    click.echo(f"\n=== {symbol} Analysis ===")
    click.echo(f"Current Price: ${analysis['current_price']:.2f}")
    click.echo(f"Price Change: ${analysis['price_change']:.2f} ({analysis['price_change_pct']:.2f}%)")
    click.echo(f"52W High: ${analysis['52_week_high']:.2f}")
    click.echo(f"52W Low: ${analysis['52_week_low']:.2f}")
    
    if analysis['current_rsi']:
        click.echo(f"RSI: {analysis['current_rsi']:.2f}")
    
    if analysis['current_volatility']:
        click.echo(f"Volatility: {analysis['current_volatility']:.2f}")


@cli.command()
@click.argument('symbols', nargs=-1, required=True)
@click.option('--period', default='1y', help='Data period')
def compare(symbols, period):
    """Compare multiple stocks."""
    analyzer = StockAnalyzer()
    click.echo(f"Comparing stocks: {', '.join(symbols)}")
    
    comparison = analyzer.compare_stocks(list(symbols), period)
    if not comparison.empty:
        click.echo("\n" + comparison.to_string(index=False))
    else:
        click.echo("No data available for comparison")


@cli.command()
@click.argument('symbol')
@click.option('--period', default='1y', help='Data period')
def plot(symbol, period):
    """Plot stock analysis charts."""
    analyzer = StockAnalyzer()
    click.echo(f"Generating plots for {symbol}...")
    analyzer.plot_stock_analysis(symbol, period)


@cli.command()
def portfolio():
    """Interactive portfolio management."""
    portfolio = Portfolio()
    
    while True:
        click.echo("\n=== Portfolio Management ===")
        click.echo("1. Buy stock")
        click.echo("2. Sell stock")
        click.echo("3. View portfolio")
        click.echo("4. View performance")
        click.echo("5. View transactions")
        click.echo("6. Exit")
        
        choice = click.prompt("Choose an option", type=int)
        
        if choice == 1:
            symbol = click.prompt("Enter stock symbol").upper()
            shares = click.prompt("Enter number of shares", type=int)
            if portfolio.buy_stock(symbol, shares):
                click.echo("Purchase successful!")
            else:
                click.echo("Purchase failed!")
        
        elif choice == 2:
            symbol = click.prompt("Enter stock symbol").upper()
            shares = click.prompt("Enter number of shares", type=int)
            if portfolio.sell_stock(symbol, shares):
                click.echo("Sale successful!")
            else:
                click.echo("Sale failed!")
        
        elif choice == 3:
            holdings = portfolio.get_holdings_summary()
            if not holdings.empty:
                click.echo("\n" + holdings.to_string(index=False))
            else:
                click.echo("No holdings found")
        
        elif choice == 4:
            performance = portfolio.get_portfolio_performance()
            click.echo(f"\n=== Portfolio Performance ===")
            click.echo(f"Initial Value: ${performance['initial_value']:.2f}")
            click.echo(f"Current Value: ${performance['current_value']:.2f}")
            click.echo(f"Total Return: ${performance['total_return']:.2f}")
            click.echo(f"Return %: {performance['return_percentage']:.2f}%")
            click.echo(f"Available Cash: ${performance['cash']:.2f}")
        
        elif choice == 5:
            transactions = portfolio.get_transaction_history()
            if not transactions.empty:
                click.echo("\n" + transactions.to_string(index=False))
            else:
                click.echo("No transactions found")
        
        elif choice == 6:
            click.echo("Goodbye!")
            break
        
        else:
            click.echo("Invalid option")


@cli.command()
@click.argument('symbol')
@click.option('--period', default='1y', help='Data period for backtesting')
@click.option('--initial-capital', default=10000.0, help='Initial capital for backtesting')
def backtest_close_open(symbol, period, initial_capital):
    """Backtest close-to-open strategy on a symbol."""
    backtester = Backtester(initial_capital=initial_capital)
    click.echo(f"Backtesting close-to-open strategy for {symbol}...")
    
    results = backtester.backtest_close_to_open(symbol, period=period)
    
    if 'error' in results:
        click.echo(f"Error: {results['error']}")
        return
    
    click.echo(f"\n=== Backtest Results for {symbol} ===")
    click.echo(f"Strategy: {results['strategy']}")
    click.echo(f"Period: {results['period']}")
    click.echo(f"Initial Capital: ${results['initial_capital']:,.2f}")
    click.echo(f"Final Value: ${results['final_value']:,.2f}")
    click.echo(f"Total Return: ${results['total_return']:,.2f}")
    click.echo(f"Return %: {results['return_percentage']:.2f}%")
    click.echo(f"Buy & Hold Return: {results['buy_hold_return']:.2f}%")
    click.echo(f"Excess Return: {results['excess_return']:.2f}%")
    click.echo(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    click.echo(f"Win Rate: {results['win_rate']:.1f}%")
    click.echo(f"Total Trades: {results['total_trades']}")


@cli.command()
@click.argument('symbols', nargs=-1, required=True)
@click.option('--period', default='1y', help='Data period for backtesting')
@click.option('--initial-capital', default=10000.0, help='Initial capital for backtesting')
def compare_backtest(symbols, period, initial_capital):
    """Compare close-to-open strategy across multiple symbols."""
    backtester = Backtester(initial_capital=initial_capital)
    click.echo(f"Comparing close-to-open strategy across: {', '.join(symbols)}")
    
    results_summary = []
    
    for symbol in symbols:
        try:
            results = backtester.backtest_close_to_open(symbol, period=period)
            
            if 'error' not in results:
                results_summary.append({
                    'Symbol': symbol,
                    'Return %': f"{results['return_percentage']:.2f}%",
                    'Buy & Hold %': f"{results['buy_hold_return']:.2f}%",
                    'Excess Return %': f"{results['excess_return']:.2f}%",
                    'Sharpe Ratio': f"{results['sharpe_ratio']:.3f}",
                    'Win Rate %': f"{results['win_rate']:.1f}%",
                    'Total Trades': results['total_trades']
                })
            else:
                click.echo(f"Error with {symbol}: {results['error']}")
        
        except Exception as e:
            click.echo(f"Error testing {symbol}: {str(e)}")
    
    if results_summary:
        click.echo(f"\n=== Strategy Comparison ===")
        # Simple table display
        headers = results_summary[0].keys()
        click.echo(" | ".join(f"{h:>12}" for h in headers))
        click.echo("-" * (13 * len(headers) + len(headers) - 1))
        
        for result in results_summary:
            click.echo(" | ".join(f"{str(v):>12}" for v in result.values()))


@cli.command()
def demo_strategy():
    """Run a demo of the close-to-open strategy with SPY and QQQ."""
    click.echo("=" * 60)
    click.echo("Close-to-Open Strategy Demo")
    click.echo("Strategy: Buy at close, sell at next open (every day)")
    click.echo("=" * 60)
    
    backtester = Backtester(initial_capital=10000.0)
    symbols = ['SPY', 'QQQ']
    
    for symbol in symbols:
        click.echo(f"\n--- Testing {symbol} ---")
        
        try:
            results = backtester.backtest_close_to_open(symbol, period="6mo")
            
            if 'error' in results:
                click.echo(f"Error: {results['error']}")
                continue
            
            click.echo(f"Return: {results['return_percentage']:.2f}% vs Buy & Hold: {results['buy_hold_return']:.2f}%")
            click.echo(f"Win Rate: {results['win_rate']:.1f}% ({results['total_trade_pairs']} trade pairs)")
            
            # Show a few recent trades
            if results['trades']:
                click.echo("Recent trades:")
                recent = results['trades'][-6:]  # Last 6 transactions
                for trade in recent:
                    click.echo(f"  {trade['date'].strftime('%Y-%m-%d')}: {trade['type']} {trade['shares']} @ ${trade['price']:.2f}")
        
        except Exception as e:
            click.echo(f"Error: {str(e)}")
            click.echo("Try installing required packages: pip install yfinance pandas numpy")


def main():
    """Main function."""
    cli()


if __name__ == '__main__':
    main()

<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Stock Investor Application - Copilot Instructions

This is a Python-based stock investment analysis and portfolio management application. When working on this project, please follow these guidelines:

## Project Structure
- `src/` - Main source code
  - `data_fetcher.py` - Stock data retrieval from various APIs
  - `portfolio.py` - Portfolio management and tracking
  - `analyzer.py` - Technical and fundamental analysis tools
  - `main.py` - CLI application entry point
- `tests/` - Unit tests using pytest
- `data/` - Data storage (CSV files, databases)
- `docs/` - Documentation and examples

## Coding Standards
- Follow PEP 8 style guidelines
- Use type hints for all function parameters and return values
- Include comprehensive docstrings for all classes and methods
- Write unit tests for new functionality
- Use logging instead of print statements for debugging

## Key Dependencies
- `yfinance` - Primary data source for stock prices
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computations
- `matplotlib/seaborn/plotly` - Data visualization
- `click` - Command-line interface
- `pytest` - Testing framework

## Financial Domain Guidelines
- Always handle API rate limits and failures gracefully
- Validate stock symbols before making API calls
- Include error handling for market data unavailability
- Consider market hours and holidays when fetching real-time data
- Implement proper risk management in portfolio operations
- Use appropriate financial calculations (annualized returns, volatility, etc.)

## Data Handling
- Cache data appropriately to minimize API calls
- Handle missing or incomplete data gracefully
- Validate data integrity before calculations
- Store transaction history for audit trails

## Security Considerations
- Never hardcode API keys or sensitive data
- Use environment variables for configuration
- Implement proper input validation for user inputs
- Sanitize financial data before display

When suggesting code improvements or new features, prioritize:
1. Data accuracy and reliability
2. Performance optimization for large datasets
3. User experience and error handling
4. Maintainable and testable code
5. Financial calculation correctness

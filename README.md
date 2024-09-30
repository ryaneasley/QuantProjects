# PortfolioGenetic Optimizer.py

# Overview:
# - Implements a Genetic Algorithm (GA) to optimize portfolio's Sharpe ratio and return.
# - Fetches monthly stock prices using yfinance.
# - Custom mutation and crossover operators ensure portfolio weights are valid.
# - Optimizes for maximum Sharpe ratio and return.

# Libraries Required:
# pip install yfinance pandas numpy deap tqdm multiprocessing

# How to Run:
# 1. Customize tickers, start_date, and end_date in the script.
# 2. Run the script:
#    python portfolio_genetic_optimizer.py

# Input:
# - Historical stock data fetched for tickers between start and end dates.

# Output:
# - Optimal portfolio based on Sharpe ratio.
# - Key metrics: Sharpe Ratio, Expected Return, Maximum Drawdown.

# BacktestPortfolio.py

# Overview:
# - Uses linear programming to optimize portfolio share allocation with ±1% margin.
# - Fetches historical prices using yfinance and handles missing data.
# - Calculates portfolio metrics (Sharpe Ratio, Sortino Ratio, Maximum Drawdown).

# Libraries Required:
# pip install yfinance pandas numpy matplotlib seaborn pulp argparse

# How to Run:
# 1. Customize tickers, weights, start_date, and end_date in the script.
# 2. Run the script with an investment parameter:
#    python portfolio_optimizer.py --investment 100000

# Input:
# - Historical stock data for tickers with target weights.
# - Target total investment amount (e.g., $100,000).

# Output:
# - Optimal share allocation, investment, and weight distribution.
# - Key metrics: Annualized Return, Sharpe Ratio, Sortino Ratio, Maximum Drawdown.
# - Visualizations: Cumulative returns, drawdown, and portfolio distribution.

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import pulp
import argparse
from multiprocessing import Pool
import sys


def fetch_historical_data(tickers, start_date, end_date):
    """
    Fetches historical adjusted close prices for given tickers and date range.

    Parameters:
    - tickers (list): List of stock tickers.
    - start_date (str): Start date in 'YYYY-MM-DD' format.
    - end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
    - historical_data (pd.DataFrame): DataFrame of adjusted close prices.
    """
    print("Fetching historical data...")
    historical_data = yf.download(tickers, start=start_date, end=end_date, threads=False)['Adj Close']
    print("Historical data fetched successfully.\n")
    return historical_data


def handle_missing_data(historical_data):
    """
    Handles missing data by forward filling and then backward filling.
    Drops any remaining missing data.

    Parameters:
    - historical_data (pd.DataFrame): DataFrame of historical prices.

    Returns:
    - historical_data (pd.DataFrame): Cleaned DataFrame with no missing data.
    """
    print("Handling missing data...")
    missing_data = historical_data.isnull().sum()
    if missing_data.any():
        print("Warning: Missing data detected for some tickers.")
        print("Tickers with missing data:\n", missing_data[missing_data > 0])

    historical_data.ffill(inplace=True)
    historical_data.bfill(inplace=True)

    # Check again
    if historical_data.isnull().any().any():
        print("Warning: Some missing data could not be filled. Dropping these tickers.")
        historical_data.dropna(axis=1, inplace=True)
    else:
        print("All missing data handled successfully.")
    print()
    return historical_data


def plot_correlation_heatmap(daily_returns):
    """
    Plots the correlation heatmap of the portfolio.

    Parameters:
    - daily_returns (pd.DataFrame): DataFrame of daily returns.
    """
    print("Plotting correlation heatmap...")
    plt.figure(figsize=(16, 14))
    sns.heatmap(daily_returns.corr(), annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title("Portfolio Correlation Heatmap")
    plt.show()
    print("Correlation heatmap displayed.\n")


def optimize_share_allocation_with_margin(valid_prices, desired_weights, margin=0.01, target_total_investment=100000):
    """
    Optimizes share allocation allowing total investment to vary within ±margin.

    Parameters:
    - valid_prices (pd.Series): Current prices of the tickers.
    - desired_weights (dict): Desired weight for each ticker.
    - margin (float): Allowed error margin (e.g., 0.01 for ±1%).
    - target_total_investment (float): Target total investment amount.

    Returns:
    - allocated_shares (dict): Number of shares allocated per ticker.
    - actual_investment (dict): Actual investment per ticker.
    - actual_weights (dict): Actual weight per ticker.
    - total_investment (float): The total investment amount.
    - success (bool): Indicator of successful optimization.
    """
    print("Defining the linear programming problem...")
    # Define the problem
    prob = pulp.LpProblem("ShareAllocation", pulp.LpMinimize)

    # Define variables: number of shares for each ticker (Continuous for fractional shares)
    share_vars = {ticker: pulp.LpVariable(f"Shares_{ticker}", lowBound=0, cat='Continuous') for ticker in valid_prices.index}

    # Define total investment variable
    total_investment = pulp.LpVariable("TotalInvestment", lowBound=0)

    # Objective: Minimize the total discrepancy between desired and actual investments
    discrepancy = {ticker: pulp.LpVariable(f"Discrepancy_{ticker}", lowBound=0) for ticker in valid_prices.index}
    prob += pulp.lpSum([discrepancy[ticker] for ticker in valid_prices.index]), "TotalDiscrepancy"

    # Define investment bounds
    lower_total_investment = target_total_investment * (1 - margin)
    upper_total_investment = target_total_investment * (1 + margin)

    # Constraint for total investment within ±margin
    prob += pulp.lpSum([share_vars[ticker] * valid_prices[ticker] for ticker in valid_prices.index]) >= lower_total_investment, "TotalInvestment_LowerBound"
    prob += pulp.lpSum([share_vars[ticker] * valid_prices[ticker] for ticker in valid_prices.index]) <= upper_total_investment, "TotalInvestment_UpperBound"

    # Constraints for each ticker's investment within ±margin of desired investment
    for ticker in valid_prices.index:
        desired_investment = desired_weights[ticker] * target_total_investment
        min_investment = desired_investment * (1 - margin)
        max_investment = desired_investment * (1 + margin)

        # Investment bounds for the ticker
        prob += share_vars[ticker] * valid_prices[ticker] >= min_investment, f"WeightBounds_Lower_{ticker}"
        prob += share_vars[ticker] * valid_prices[ticker] <= max_investment, f"WeightBounds_Upper_{ticker}"

        # Discrepancy constraints
        actual_investment = share_vars[ticker] * valid_prices[ticker]
        prob += actual_investment - desired_investment <= discrepancy[ticker], f"DiscrepancyPos_{ticker}"
        prob += desired_investment - actual_investment <= discrepancy[ticker], f"DiscrepancyNeg_{ticker}"

    print("Solving the optimization problem...")
    # Solve the problem with solver verbosity enabled
    prob.solve(pulp.PULP_CBC_CMD(msg=True))

    print(f"Optimization Status: {pulp.LpStatus[prob.status]}\n")

    # Check if the solution is optimal
    if pulp.LpStatus[prob.status] == 'Optimal':
        print("Optimal solution found.")
        allocated_shares = {ticker: pulp.value(share_vars[ticker]) for ticker in valid_prices.index}
        actual_investment = {ticker: allocated_shares[ticker] * valid_prices[ticker] for ticker in valid_prices.index}
        total_investment_value = sum(actual_investment.values())
        actual_weights = {ticker: actual_investment[ticker] / total_investment_value for ticker in valid_prices.index}
        print("Share allocation completed successfully.\n")
        return allocated_shares, actual_investment, actual_weights, total_investment_value, True
    else:
        print("No optimal solution found.")
        print("Optimization failed to find a valid share allocation within constraints.\n")
        return None, None, None, None, False


def plot_cumulative_and_drawdown(cumulative_returns, drawdown):
    """
    Plots cumulative returns and drawdown in two subplots.

    Parameters:
    - cumulative_returns (pd.Series): Cumulative returns of the portfolio.
    - drawdown (pd.Series): Drawdown series of the portfolio.
    """
    print("Plotting cumulative returns and drawdown...")
    plt.figure(figsize=(14, 10))

    # Cumulative Returns Plot
    plt.subplot(2, 1, 1)
    plt.plot(cumulative_returns.index, cumulative_returns, label='Cumulative Returns', color='blue')
    plt.title("Portfolio Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend(loc='upper left')
    plt.grid(True)

    # Drawdown Plot
    plt.subplot(2, 1, 2)
    plt.plot(drawdown.index, drawdown, label='Drawdown', color='red')
    plt.title("Portfolio Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend(loc='upper left')
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    print("Cumulative returns and drawdown plots displayed.\n")


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Portfolio Optimization Tool")
    parser.add_argument(
        '-i', '--investment',
        type=float,
        default=100000.0,  # Input Target Investment Amount
        help="Target total investment amount in USD (e.g., 100000)"
    )
    args = parser.parse_args()
    target_total_investment = args.investment

    # ----------------------------- User Inputs -----------------------------

    # Define your portfolio tickers with corresponding weights
    portfolio = {
    'INSM': 0.0184,
    'STAA': 0.0460,
    'SHEL': 0.0441,
    'RIOCF': 0.0028,
    'FDP': 0.0030,
    'PRGO': 0.0165,
    'OPCH': 0.0021,
    'HMC': 0.0021,
    'LDOS': 0.0457,
    'CQP': 0.0456,
    'PAHC': 0.0453,
    'GLW': 0.0272,
    'VLO': 0.0028,
    'GATX': 0.0020,
    'BURL': 0.0020,
    'PRU': 0.0032,
    'MCY': 0.0405,
    'ADI': 0.0029,
    'IRM': 0.0215,
    'LEN': 0.0020,
    'TDC': 0.0447,
    'AB': 0.0245,
    'DVN': 0.0040,
    'PODD': 0.0020,
    'GNK': 0.0020,
    'ALGN': 0.0020,
    'KR': 0.0034,
    'ADM': 0.0027,
    'IESC': 0.0116,
    'CIM': 0.0147,
    'REG': 0.0357,
    'AAGIY': 0.0309,
    'CIEN': 0.0392,
    'RNR': 0.0413,
    'NVGS': 0.0080,
    'MRVL': 0.0023,
    'STC': 0.0025,
    'FRT': 0.0409,
    'IRT': 0.0422,
    'NVDA': 0.0375,
    'ROG': 0.0031,
    'BYRN': 0.0465,
    'FLEX': 0.0023,
    'CALM': 0.0020,
    'GLNG': 0.0463,
    'AX': 0.0198,
    'BAC': 0.0025,
    'ENB': 0.0374,
    'LPX': 0.0024,
    'EPAC': 0.0037,
    }
    # Verify that desired weights sum to 1 within a ±1% margin
    total_weight = sum(portfolio.values())
    if not 0.99 <= total_weight <= 1.01:
        print(f"Warning: Total desired weights sum to {total_weight:.4f}, which is outside the ±1% margin. Normalizing to 1.")
        portfolio = {ticker: weight / total_weight for ticker, weight in portfolio.items()}
    else:
        print(f"Total desired weights sum to {total_weight:.4f}, within the ±1% margin. No normalization needed.")
    print()

    # Specify the date range for historical data
    start_date = '2019-01-01'
    end_date = '2024-09-28'

    # Define the export path for the CSV file
    export_path = r"C:\Users\ryane\Downloads\minimum_portfolio_shares.csv"

    # Risk-free rate (annual)
    risk_free_rate = 0.05  # 5%

    # Margin for total investment
    margin = 0.01  # 1%

    # ----------------------------- Data Fetching -----------------------------

    tickers = list(portfolio.keys())
    historical_data = fetch_historical_data(tickers, start_date, end_date)
    historical_data = handle_missing_data(historical_data)

    if historical_data.empty:
        print("No historical data available after handling missing data. Exiting.")
        sys.exit()

    # ----------------------------- Portfolio Analysis -----------------------------

    print("Analyzing portfolio...")
    # Calculate daily returns
    daily_returns = historical_data.pct_change().dropna()

    # Calculate the correlation matrix
    corr_matrix = daily_returns.corr()

    # Plot the correlation heatmap
    plot_correlation_heatmap(daily_returns)

    # ----------------------------- Investment Calculation -----------------------------

    # Fetch current prices for the tickers (1-day period)
    print("Fetching current prices...")
    try:
        current_prices = yf.download(tickers, period='1d', threads=False)['Adj Close'].iloc[0]
        print("Current prices fetched successfully.\n")
    except Exception as e:
        print(f"Error fetching current prices: {e}")
        sys.exit()

    # Handle missing price data
    valid_prices = current_prices.dropna()

    if valid_prices.empty:
        print("No valid prices found for the provided tickers. Exiting.")
        sys.exit()

    print("Valid prices identified for the following tickers:")
    print(valid_prices, "\n")

    # Adjust portfolio weights for valid tickers
    adjusted_weights = {ticker: portfolio[ticker] for ticker in valid_prices.index}
    adjusted_weights_total = sum(adjusted_weights.values())
    adjusted_weights = {ticker: weight / adjusted_weights_total for ticker, weight in adjusted_weights.items()}

    print("Adjusted portfolio weights for valid tickers:")
    print(adjusted_weights, "\n")

    # Start Optimization
    print("Starting share allocation optimization...")
    # Optimize share allocation using PuLP with margin
    allocated_shares, actual_investment, actual_weights, total_investment, success = optimize_share_allocation_with_margin(
        valid_prices, adjusted_weights, margin, target_total_investment
    )

    print(f"Optimization Success: {success}\n")

    if not success:
        print("Optimization failed to find a valid share allocation within constraints.")
        print("Consider relaxing weight constraints, increasing total investment, or adjusting desired weights.")
        sys.exit()
    else:
        print("Creating DataFrame with allocated shares...")

    # Create a DataFrame to display shares, prices, and investments
    shares_per_stock_df = pd.DataFrame({
        'Price (USD)': valid_prices.values,
        'Desired Weight': [adjusted_weights[ticker] for ticker in valid_prices.index],
        'Actual Weight': [actual_weights[ticker] for ticker in valid_prices.index],
        'Shares': [allocated_shares[ticker] for ticker in valid_prices.index],
        'Investment (USD)': [actual_investment[ticker] for ticker in valid_prices.index]
    }, index=valid_prices.index)

    print("DataFrame created successfully.\n")
    print(shares_per_stock_df)
    print(f"\nTotal Investment Allocated: ${total_investment:,.2f}\n")

    # Save the DataFrame to CSV
    try:
        shares_per_stock_df.to_csv(export_path)
        print(f"Data exported successfully to {export_path}\n")
    except Exception as e:
        print(f"Error exporting data to CSV: {e}")
        sys.exit()

    # ----------------------------- Portfolio Metrics -----------------------------

    print("Calculating portfolio metrics...")
    # Calculate portfolio returns based on actual weights
    portfolio_returns = daily_returns[list(valid_prices.index)].dot(list(actual_weights.values()))

    # Calculate realized return (total return over the date range)
    realized_return = (1 + portfolio_returns).prod() - 1

    # Calculate the number of years in the date range and the number of months
    start_datetime = pd.to_datetime(start_date)
    end_datetime = pd.to_datetime(end_date)
    n_days = (end_datetime - start_datetime).days
    n_years = n_days / 365
    n_months = (end_datetime.year - start_datetime.year) * 12 + (end_datetime.month - start_datetime.month)

    # Calculate annualized return
    annualized_return = (1 + realized_return) ** (1 / n_years) - 1

    # Calculate realized volatility (standard deviation of daily returns)
    realized_volatility = portfolio_returns.std()

    # Calculate annualized volatility
    annualized_volatility = realized_volatility * np.sqrt(252)

    # Calculate Sortino Ratio
    downside_returns = portfolio_returns[portfolio_returns < 0]
    if len(downside_returns) > 0:
        downside_deviation = downside_returns.std() * np.sqrt(252)
        sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation
    else:
        sortino_ratio = np.nan  # Undefined if no downside returns

    # Calculate Sharpe Ratio
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

    # Calculate maximum drawdown
    cumulative_returns = (1 + portfolio_returns).cumprod()
    running_max = cumulative_returns.cummax()
    drawdown = (cumulative_returns - running_max) / running_max
    max_drawdown = drawdown.min()

    print("Portfolio metrics calculated successfully.\n")

    # Print portfolio metrics in the desired order
    print("Portfolio Metrics:")
    print(f"Realized Return ({n_months} months): {realized_return:.2%}")
    print(f"Annualized Return: {annualized_return:.2%}")
    print(f"Annualized Volatility: {annualized_volatility:.2%}")
    print(f"Realized Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Realized Sortino Ratio: {sortino_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}\n")

    # ----------------------------- Visualization -----------------------------

    print("Generating visualizations...")

    # Plot the fetched prices for verification
    print("Plotting fetched stock prices...")
    plt.figure(figsize=(12, 8))
    valid_prices.plot(kind='bar', color='skyblue')
    plt.title("Fetched Stock Prices")
    plt.ylabel("Price (USD)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    print("Fetched stock prices plotted.\n")

    # Plot the pie chart for actual investments
    print("Plotting portfolio investment distribution pie chart...")
    plt.figure(figsize=(16, 14))  # Increased size for better readability
    plt.pie(
        shares_per_stock_df['Investment (USD)'],
        labels=shares_per_stock_df.index,
        autopct='%1.1f%%',
        startangle=90
    )
    plt.title("Portfolio Investment Distribution")

    # Create a legend showing ticker, shares, and dollar investment
    legend_labels = [
        f"{ticker}: {shares:.2f} shares, ${investment:,.2f}"
        for ticker, shares, investment in zip(
            shares_per_stock_df.index,
            shares_per_stock_df['Shares'],
            shares_per_stock_df['Investment (USD)']
        )
    ]

    # Generate unique colors for the legend
    colors = plt.cm.tab20.colors
    patches = [
        mpatches.Patch(color=colors[i % len(colors)], label=legend_labels[i])
        for i in range(len(shares_per_stock_df))
    ]

    plt.legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.tight_layout()
    plt.show()
    print("Portfolio investment distribution pie chart plotted.\n")

    # Plotting Cumulative Returns and Drawdown
    plot_cumulative_and_drawdown(cumulative_returns, drawdown)


if __name__ == "__main__":
    main()

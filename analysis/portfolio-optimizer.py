import argparse
import json
import pandas as pd
import pulp
import logging

# DO NOT DELETE THIS COMMENT
#  python3 portfolio-optimizer.py mother-iggy --budget 80000 --holding_period_months 6


def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_json_files(file_paths):
    price_data = {}
    for file_path in file_paths:
        try:
            with open(file_path, "r") as file:
                data = json.load(file)
                timestamps = [entry[0] for entry in data]
                prices = [entry[1] for entry in data]
                asset_name = file_path.split("/")[-1].split("-price-data")[0]
                price_data[asset_name] = pd.Series(
                    prices, index=pd.to_datetime(timestamps, unit="ms")
                )
            logging.info(f"Loaded data for {asset_name} from {file_path}")
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
            continue
    return pd.DataFrame(price_data)


def calculate_metrics(price_data, holding_period_months):
    trading_days_per_month = 21  # Approximate number of trading days in a month
    holding_period_days = holding_period_months * trading_days_per_month

    logging.info(
        f"Calculating metrics for a holding period of {holding_period_days} days"
    )

    # Calculate daily returns
    returns = price_data.pct_change().dropna()

    # Calculate expected returns (mean of daily returns)
    expected_returns = (
        returns.mean() * holding_period_days
    )  # Adjust for the holding period

    logging.info("Calculated expected returns")

    return expected_returns


def optimize_portfolio(
    expected_returns, budget, asset_names, min_investment_ratio, diversification
):
    num_assets = len(expected_returns)

    logging.info(f"Starting optimization with a budget of ${budget}")

    # Create a LP problem
    prob = pulp.LpProblem("Portfolio_Optimization", pulp.LpMaximize)

    # Decision variables
    investment = pulp.LpVariable.dicts(
        "investment", range(num_assets), lowBound=0, cat="Continuous"
    )

    # Objective function: Maximize returns
    prob += pulp.lpSum([expected_returns[i] * investment[i] for i in range(num_assets)])

    # Constraint: Budget
    prob += (
        pulp.lpSum([investment[i] for i in range(num_assets)]) <= budget,
        "BudgetConstraint",
    )

    if diversification:
        logging.info("Applying diversification constraints")
        # Diversification Constraint: Minimum investment per asset
        min_investment_per_asset = min_investment_ratio * budget
        for i in range(num_assets):
            prob += (
                investment[i] >= min_investment_per_asset,
                f"MinInvestmentConstraint_{i}",
            )

    prob.solve()

    # Output the results
    if pulp.LpStatus[prob.status] == "Optimal":
        logging.info("Optimization completed successfully")
        total_investment = 0
        for i, v in enumerate(prob.variables()):
            logging.info(f"Investment in {asset_names[i]} = ${v.varValue:.2f} USD")
            total_investment += v.varValue
        expected_return = pulp.value(prob.objective)
        total_ending_amount = total_investment + expected_return
        investment_ratio = total_ending_amount / total_investment
        logging.info("-" * 30)
        logging.info(f"Total Initial Investment: ${total_investment:.2f} USD")
        logging.info(f"Total Expected Return: ${expected_return:.2f} USD")
        logging.info(
            f"Total Ending Amount: ${total_ending_amount:.2f} USD ({investment_ratio:.2f}x)"
        )
    else:
        logging.error("No optimal solution found")


def main():
    setup_logging()

    parser = argparse.ArgumentParser(description="Optimize cryptocurrency portfolio.")
    parser.add_argument(
        "coins", metavar="C", type=str, nargs="+", help="List of cryptocurrency coins"
    )
    parser.add_argument(
        "--budget", type=float, required=True, help="Total investment budget"
    )
    parser.add_argument(
        "--holding_period_months",
        type=int,
        required=True,
        help="Holding period in months",
    )
    parser.add_argument(
        "--min_investment_ratio",
        type=float,
        default=0.05,
        help="Minimum investment ratio per asset (default: 0.05)",
    )
    parser.add_argument(
        "--diversify",
        action="store_true",
        help="Enable diversification mode (default: disabled for single-asset allocation)",
    )

    args = parser.parse_args()

    logging.info("Loading price data")
    file_paths = [f"../data/generated/{coin}-price-data.json" for coin in args.coins]

    price_data = load_json_files(file_paths)

    logging.info("Calculating expected returns")
    expected_returns = calculate_metrics(price_data, args.holding_period_months)

    logging.info("Running portfolio optimization")
    optimize_portfolio(
        expected_returns.values,
        args.budget,
        args.coins,
        args.min_investment_ratio,
        args.diversify,
    )


if __name__ == "__main__":
    main()

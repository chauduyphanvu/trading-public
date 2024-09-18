import argparse
import json
import logging

import numpy as np
from matplotlib import pyplot as plt


# python3 kelly-criterion.py -f ../data/src/coingecko-market_chart-data-hourly.json -c gme -capital 100 -current_investment 0 -i 4 -w 10


# Load stock price data from JSON file
def load_data(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def calculate_volatility(prices, window_size=14):
    returns = np.diff(prices) / prices[:-1]
    volatility = np.std(returns[-window_size:])
    return volatility


def calculate_dynamic_kelly_fraction(returns, volatility, base_kelly_fraction=1.0):
    mean_return = np.mean(returns)
    variance_return = np.var(returns)
    kelly_fraction = mean_return / variance_return

    # Adjust Kelly fraction based on volatility
    if volatility > 0.05:  # Example threshold for high volatility
        kelly_fraction *= 0.5  # Reduce by half in high volatility
    elif volatility < 0.01:  # Example threshold for low volatility
        kelly_fraction *= 1.5  # Increase by 50% in low volatility

    # Cap the Kelly fraction at 100%
    return min(kelly_fraction, base_kelly_fraction)


# Calculate daily returns
def calculate_daily_returns(prices):
    returns = []
    for i in range(1, len(prices)):
        previous_price = prices[i - 1][1]
        current_price = prices[i][1]
        daily_return = (current_price - previous_price) / previous_price
        returns.append(daily_return)
    return returns


# Calculate Kelly fraction
def calculate_kelly_fraction(returns):
    mean_return = np.mean(returns)
    variance_return = np.var(returns)
    kelly_fraction = mean_return / variance_return
    return kelly_fraction


# Decide total investment amount based on Kelly criterion
def decide_total_investment(kelly_fraction, total_capital, current_investment):
    kelly_fraction = min(kelly_fraction, 1.0)  # Cap the Kelly fraction at 100%
    total_investment = kelly_fraction * (total_capital - current_investment)
    return max(0, total_investment)  # Ensure non-negative investment


# Calculate moving average
def calculate_moving_average(prices, window_size):
    moving_averages = []
    for i in range(len(prices)):
        if i < window_size:
            moving_averages.append(
                None
            )  # Not enough data to calculate the moving average
        else:
            window = prices[i - window_size : i]
            moving_average = np.mean([price[1] for price in window])
            moving_averages.append(moving_average)
    return moving_averages


# Divide total investment into DCA intervals and suggest prices
def calculate_dca_intervals(total_investment, prices, moving_averages, dca_intervals):
    interval_investment = total_investment / dca_intervals
    interval_prices = []

    # Calculate the average price for each interval using moving averages
    interval_length = len(prices) // dca_intervals
    for i in range(dca_intervals):
        start_index = i * interval_length
        end_index = (i + 1) * interval_length
        interval_average_prices = [
            moving_averages[j]
            for j in range(start_index, end_index)
            if moving_averages[j] is not None
        ]
        if interval_average_prices:
            average_price = np.mean(interval_average_prices)
        else:
            average_price = 0  # Default to 0 if no valid moving average is found
        interval_prices.append((interval_investment, average_price))

    return interval_prices


def main():
    # Argument parsing and data loading (unchanged)
    parser = argparse.ArgumentParser(
        description="Calculate Kelly Criterion for investment asset."
    )
    parser.add_argument(
        "-f",
        "--filename",
        type=str,
        required=True,
        help="Path to the JSON file with stock data",
    )
    parser.add_argument(
        "-c",
        "--coin_id",
        type=str,
        required=True,
        help="Coin ID to analyze (e.g., 'gme')",
    )
    parser.add_argument(
        "-capital",
        type=float,
        default=100,
        help="Total capital available for investment (default is 100)",
    )
    parser.add_argument(
        "-current_investment",
        type=float,
        default=0,
        help="Current investment in the stock (default is 0)",
    )
    parser.add_argument(
        "-i",
        "--intervals",
        type=int,
        default=4,
        help="Number of intervals to spread the DCA investments over (default is 4)",
    )
    parser.add_argument(
        "-w",
        "--window_size",
        type=int,
        default=14,
        help="Window size for moving average calculation (default is 14)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    data = load_data(args.filename)
    prices = data["data"][args.coin_id]["prices"]
    closing_prices = np.array([price[1] for price in prices])

    moving_averages = calculate_moving_average(prices, args.window_size)

    valid_prices = [
        price for price, ma in zip(prices, moving_averages) if ma is not None
    ]
    valid_moving_averages = [ma for ma in moving_averages if ma is not None]

    if len(valid_prices) < 2:
        logging.info(
            f"Not enough data for the moving average calculation with window size {args.window_size}."
        )
        return

    daily_returns = calculate_daily_returns(valid_prices)
    volatility = calculate_volatility(closing_prices, window_size=args.window_size)
    kelly_fraction = calculate_dynamic_kelly_fraction(daily_returns, volatility)

    total_investment = decide_total_investment(
        kelly_fraction, args.capital, args.current_investment
    )

    dca_intervals = calculate_dca_intervals(
        total_investment, valid_prices, valid_moving_averages, args.intervals
    )

    logging.info(
        f"Calculating Kelly Criterion for asset with ID `{args.coin_id}` to determine DCA strategy."
    )
    logging.info(f"You have ${args.capital:.2f} available for investment.")
    logging.info(f"Moving Average Window Size: {args.window_size} days")
    logging.info(f"Volatility: {volatility:.2%}")
    logging.info(f"Kelly Fraction: {kelly_fraction:.2%}")
    logging.info(f"Total Recommended Investment: ${total_investment:.2f}")
    logging.info(
        f"Recommended DCA amounts and price entry over {args.intervals} intervals:"
    )

    # Prepare data for the table
    summary_data = [
        ["Available Capital", f"${args.capital:.2f}"],
        ["Moving Average Window Size", f"{args.window_size} days"],
        ["Volatility", f"{volatility:.2%}"],
        ["Kelly Fraction", f"{kelly_fraction:.2%}"],
        ["Total Recommended Investment", f"${total_investment:.2f}"],
    ]

    table_data = [["Interval", "Investment ($)", "Average Price ($)"]]
    for i, (investment, price) in enumerate(dca_intervals):
        table_data.append([f"{i + 1}", f"${investment:.2f}", f"${price:.4f}"])

    # Create a new figure
    fig, ax = plt.subplots(figsize=(8, len(summary_data) + len(table_data) * 0.5))

    # Hide all axes and frame
    ax.axis("tight")
    ax.axis("off")

    # Combine summary data and DCA data for the table
    combined_data = (
        [["Summary", "Values", ""]]
        + [row + [""] for row in summary_data]
        + [["", "", ""]]
        + table_data
    )

    # Create table
    table = plt.table(cellText=combined_data, cellLoc="center", loc="center")

    # Adjust table scale and appearance
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.2)

    # Customize header and empty row styles
    for key, cell in table.get_celld().items():
        if key[0] == 0:  # First header row for summary
            cell.set_fontsize(12)
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#40466e")
            cell.set_text_props(color="w")
        elif key[0] == len(summary_data) + 1:  # Separator row
            cell.set_facecolor("white")
        elif key[0] == len(summary_data) + 2:  # DCA table header row
            cell.set_fontsize(12)
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#40466e")
            cell.set_text_props(color="w")

    # Display the table
    plt.title(f"{args.coin_id.upper()} | DCA Strategy with Kelly Criterion")
    filename = f"../data/generated/{args.coin_id.upper()}-kelly-dca.png"

    # Save the table as a PNG
    plt.savefig(filename, bbox_inches="tight")


if __name__ == "__main__":
    main()

"""
Simulate crypto asset price paths using Jump-Diffusion (Merton Model) and Monte Carlo.

This script uses historical price data to estimate the parameters of the Jump-Diffusion model (Merton Model) and then
simulates price paths under different market conditions (bullish, bearish, neutral) using Monte Carlo simulation.

The Jump-Diffusion model is an extension of the Geometric Brownian Motion model that incorporates jumps in asset prices.

Example:
    python3 jump-diffusion.py -f ../data/src/gme-1-hour-2019-06-24-to-2024-06-21.json -c gme -m neutral -t 200 -n 5000
"""

import argparse
import itertools
import json
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [Jump Diffusion] %(message)s",
)


def load_json_data(file_path):
    logging.info(f"Loading data from {file_path}.")

    with open(file_path, "r") as file:
        return json.load(file)


def extract_prices(data, coin_id):
    logging.info(f"Extracting price data for {coin_id}.")

    prices_data = data["data"][coin_id]["prices"]
    timestamps = [entry[0] for entry in prices_data]
    prices = [entry[1] for entry in prices_data]

    return timestamps, prices


def calculate_jump_diffusion_parameters(prices):
    logging.info("Calculating jump diffusion parameters.")

    log_returns = np.diff(np.log(prices))
    mu = np.mean(log_returns)
    sigma = np.std(log_returns)

    return mu, sigma


def simulate_jump_diffusion_paths(
    S0, mu, sigma, lambda_, mu_j, sigma_j, T, dt, num_simulations, dynamic_threshold
):
    price_paths = []
    for _ in range(num_simulations):
        prices = [S0]
        for t in range(1, T + 1):
            S_prev = prices[-1]
            dW = np.random.normal(0, np.sqrt(dt))
            jump = 0

            if np.random.rand() < lambda_ * dt:  # Poisson process for jumps
                jump = np.random.normal(mu_j, sigma_j)

            S_next = S_prev * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW + jump)

            prices.append(S_next)

        if all(price < dynamic_threshold for price in prices):
            price_paths.append(prices)

    return price_paths


def plot_simulation(price_paths, coin_id, param_combination, color, market_condition):
    logging.info(f"Plotting simulated price paths for params {param_combination}.")

    # Reduce the number of paths plotted for clarity
    max_paths_to_plot = 50
    paths_to_plot = price_paths[:max_paths_to_plot]

    all_prices = []
    for path in paths_to_plot:
        plt.plot(
            path,
            color=color,
            alpha=0.05,  # Further reduce transparency to avoid clutter
            label=f"Params: {param_combination}" if path == paths_to_plot[0] else "",
        )
        all_prices.extend(path)

    def y_format(y, _):
        return f"{y:.4f}"  # Adjust to show up to 4 decimal places

    plt.gca().yaxis.set_major_formatter(FuncFormatter(y_format))
    plt.title(
        f"{coin_id.upper()} | Simulated Price Paths (Jump Diffusion) | Assuming {market_condition.lower()} conditions"
    )
    plt.xlabel("Steps")
    plt.ylabel("Price (USD)")

    if len(all_prices) == 0:
        print("Error: No price data available for plotting.")
        return

    # Plot the mean path
    mean_path = np.mean(paths_to_plot, axis=0)
    plt.plot(mean_path, color="blue", linewidth=2, label="Mean Path")

    plt.ylim(
        bottom=0, top=0.5
    )  # Adjust this range as needed to better visualize the paths


def main():
    parser = argparse.ArgumentParser(
        description="Simulate crypto asset price paths using Jump-Diffusion (Merton Model) and Monte Carlo."
    )
    parser.add_argument(
        "-c",
        "--coin",
        type=str,
        required=True,
        help="The coin ID for the cryptocurrency.",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=True,
        help="The JSON file path containing price data.",
    )
    parser.add_argument(
        "-m",
        "--market",
        type=str,
        required=True,
        choices=["bullish", "bearish", "neutral"],
        help="The market condition (bullish, bearish, neutral).",
    )
    parser.add_argument(
        "-t",
        "--time_period",
        type=int,
        default=200,
        help="The time period for the simulation.",
    )

    parser.add_argument(
        "-n",
        "--num_simulations",
        type=int,
        default=5000,
        help="The number of simulations to run.",
    )
    args = parser.parse_args()

    data = load_json_data(args.file)
    timestamps, prices = extract_prices(data, args.coin)
    mu, sigma = calculate_jump_diffusion_parameters(prices)

    S0 = prices[-1]  # starting price (most recent price)

    T = args.time_period
    dt = 1
    num_simulations = args.num_simulations

    logging.info(
        f"Starting price: {S0}. Time period: {T} time units. Time step: {dt}. # of simulations: {num_simulations}."
    )

    # Set a threshold based on the 99th percentile of historical prices
    dynamic_threshold = np.percentile(prices, 99)
    logging.info(f"Dynamic threshold (99th percentile): {dynamic_threshold}")

    # Ensure the threshold is not below the historical ATH
    historical_ath = max(prices)
    dynamic_threshold = max(dynamic_threshold, historical_ath)
    logging.info(f"Adjusted dynamic threshold considering ATH: {dynamic_threshold}")

    # Define parameter combinations for different market conditions
    market_conditions = {
        "bullish": {
            "lambda_values": [0.1, 0.15, 0.2],
            "mu_j_values": [0.05, 0.1, 0.15],
            "sigma_j_values": [0.1, 0.15, 0.2],
        },
        "bearish": {
            "lambda_values": [0.1, 0.2, 0.3],
            "mu_j_values": [-0.2, -0.15, -0.1],
            "sigma_j_values": [0.2, 0.3, 0.4],
        },
        "neutral": {
            "lambda_values": [0.1, 0.2, 0.3],
            "mu_j_values": [-0.05, 0.0, 0.05],
            "sigma_j_values": [0.15, 0.2, 0.25],
        },
    }

    selected_market = args.market
    params = market_conditions[selected_market]
    lambda_values = params["lambda_values"]
    mu_j_values = params["mu_j_values"]
    sigma_j_values = params["sigma_j_values"]

    param_combinations = list(
        itertools.product(lambda_values, mu_j_values, sigma_j_values)
    )
    colors = plt.cm.viridis(np.linspace(0, 1, len(param_combinations)))

    plt.figure(figsize=(14, 8))

    num_workers = 50

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for color, (lambda_, mu_j, sigma_j) in zip(colors, param_combinations):
            futures.append(
                executor.submit(
                    simulate_jump_diffusion_paths,
                    S0,
                    mu,
                    sigma,
                    lambda_,
                    mu_j,
                    sigma_j,
                    T,
                    dt,
                    num_simulations,
                    dynamic_threshold,
                )
            )

        for future, color, param_combination in zip(
            as_completed(futures), colors, param_combinations
        ):
            price_paths = future.result()
            if len(price_paths) == 0:
                logging.warning(
                    f"Warning: No price paths generated for params {param_combination}."
                )
                continue
            plot_simulation(
                price_paths, args.coin, param_combination, color, selected_market
            )

    plt.show()


if __name__ == "__main__":
    main()

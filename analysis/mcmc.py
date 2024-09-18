"""
This script performs Markov Chain Monte Carlo (MCMC) simulations for price prediction. It reads a JSON file containing
historical price data, creates a transition matrix, and simulates future price paths using the transition matrix. The
script allows for stress testing by introducing shocks at specific steps in the simulation. The results are visualized
and saved to a PNG file, which is also uploaded to an S3 bucket for archival.

The script takes the following arguments:
    -f/--file: The path to the JSON file containing historical price data.
    -c/--asset_id: The ID of the asset or coin for which the simulation is performed.
    -p/--start_price: The starting price for the simulation (default: latest price).
    -steps: The number of steps to simulate.
    -runs: The number of simulation runs to perform.
    -sims: The number of simulations per run.
    -shock_steps: Comma-separated string of steps at which to introduce shocks.
    -shock_factors: Comma-separated string of factors by which to multiply the transition probabilities when a shock is introduced.

Example usage:
    python3 mcmc.py -f ../data/src/coingecko-market_chart-data-hourly.json -c mother-iggy -p latest -steps 200 -runs 100 -sims 1000 -shock_steps "1" -shock_factors "1"
"""

import argparse
import json
import logging
import sys
from collections import defaultdict, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

import boto3
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def determine_time_unit(json_file_path):
    if "30-minute" in json_file_path.lower():
        return "30-Minute"
    if "15-minute" in json_file_path.lower():
        return "15-Minute"
    elif "10-minute" in json_file_path.lower():
        return "10-Minute"
    elif "5-minute" in json_file_path.lower():
        return "5-Minute"
    elif "hour" in json_file_path.lower():
        return "1-Hour"
    elif "day" or "daily" in json_file_path.lower():
        return "1-Day"
    else:
        return "Unknown"


def load_prices_from_json(file_path, asset_id):
    with open(file_path, "r") as f:
        data = json.load(f)
    prices = [entry[1] for entry in data["data"][asset_id]["prices"]]
    return prices


def create_transition_matrix(prices):
    transitions = defaultdict(lambda: defaultdict(int))
    for current_price, next_price in zip(prices[:-1], prices[1:]):
        transitions[current_price][next_price] += 1.0
    transition_matrix = {}
    for current_price, next_prices in transitions.items():
        total_transitions = sum(next_prices.values())
        transition_matrix[current_price] = {
            next_price: count / total_transitions
            for next_price, count in next_prices.items()
        }
    return transition_matrix


def normalize_transition_matrix(transition_matrix, price_range):
    normalized_matrix = {}
    for current_price in price_range:
        next_prices = transition_matrix.get(current_price, {})
        total_prob = sum(next_prices.values())
        if total_prob > 0:
            normalized_matrix[current_price] = {
                price: prob / total_prob for price, prob in next_prices.items()
            }
        else:
            normalized_matrix[current_price] = {
                price: 1 / len(price_range) for price in price_range
            }
    return normalized_matrix


def monte_carlo_simulation(
    transition_matrix,
    price_range,
    start_price,
    steps,
    num_simulations=1000,
    shock_steps=None,
    shock_factors=None,
):
    n = len(price_range)
    P = np.zeros((n, n))
    for i, current_price in enumerate(price_range):
        for j, next_price in enumerate(price_range):
            P[i, j] = transition_matrix.get(current_price, {}).get(next_price, 0.0)
    start_index = price_range.index(start_price)
    simulations = []
    path_counter = Counter()
    shock_steps = shock_steps or []
    shock_factors = shock_factors or []

    for _ in range(num_simulations):
        prices = [start_price]
        current_index = start_index
        for step in range(steps):
            if step in shock_steps:
                shock_factor = shock_factors[shock_steps.index(step)]
                next_index = np.random.choice(n, p=P[current_index]) * shock_factor
                next_index = min(
                    max(0, int(next_index)), n - 1
                )  # Ensure index is within bounds
            else:
                next_index = np.random.choice(n, p=P[current_index])
            next_price = price_range[next_index]
            prices.append(next_price)
            current_index = next_index
        simulations.append(prices)
        path_counter[tuple(prices)] += 1
    return simulations, path_counter


def aggregate_simulations(
    num_runs,
    transition_matrix,
    price_range,
    start_price,
    steps,
    num_simulations=1000,
    max_workers=None,
    shock_steps=None,
    shock_factors=None,
):
    aggregated_path_counter = Counter()
    all_simulations = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                monte_carlo_simulation,
                transition_matrix,
                price_range,
                start_price,
                steps,
                num_simulations,
                shock_steps,
                shock_factors,
            )
            for _ in range(num_runs)
        ]
        for future in tqdm(
            as_completed(futures), total=num_runs, desc="Running simulations"
        ):
            simulations, path_counter = future.result()
            all_simulations.extend(simulations)
            aggregated_path_counter.update(path_counter)
    top_10_paths = [
        list(path) for path, count in aggregated_path_counter.most_common(10)
    ]
    return aggregated_path_counter, all_simulations, top_10_paths


def visualize_aggregated_simulations(
    top_10_paths,
    steps,
    asset_id,
    num_runs,
    num_simulations_per_run,
    time_unit,
    config_file,
):
    plt.figure(figsize=(15, 10))

    colors = plt.cm.get_cmap("tab10", len(top_10_paths)).colors

    for idx, path in enumerate(top_10_paths):
        plt.plot(
            path,
            color=colors[idx],
            linewidth=2,
            label=f"Top Path {idx + 1}",
        )

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    plt.xlabel(
        f"Steps (1 {time_unit.lower()} each)",
        color="blue",
        fontweight="bold",
        fontsize=12,
    )
    plt.ylabel("Price (USD)", color="blue", fontweight="bold")
    plt.title(
        f"{asset_id.upper()} | Markov Chain Monte Carlo (MCMC) Simulation of Future Price Paths\n"
        f"Generated on: {timestamp}.\n"
        f"{num_runs} runs. {num_simulations_per_run} simulations per run. Showing 10 most likely paths.",
        fontsize=16,
        color="purple",
    )
    plt.legend()

    file_path = f"../data/generated/{asset_id.upper()}-mcmc-{timestamp}-{steps}-{time_unit.replace(' ', '-')}.png"
    plt.savefig(file_path)

    s3_key = f"Trading/mcmc/{asset_id}-mcmc-{timestamp}-{steps}-{time_unit.replace(' ', '-')}.png"
    upload_to_s3(file_path, s3_key, config_file)


def save_results_to_json(aggregated_path_counter, top_10_paths, file_path):
    results = {
        "aggregated_path_counter": [
            {"path": list(path), "count": count}
            for path, count in aggregated_path_counter.items()
        ],
        "top_10_paths": top_10_paths,
    }
    with open(file_path, "w") as f:
        json.dump(results, f)


def log_transform_prices(prices):
    return np.log(prices)


def exp_transform_prices(log_prices):
    return np.exp(log_prices)


def upload_to_s3(file_path, s3_key, config_file):
    with open(config_file, "r") as f:
        credentials = json.load(f)["s3"]

    s3 = boto3.client(
        "s3",
        aws_access_key_id=credentials["aws_access_key_id"],
        aws_secret_access_key=credentials["aws_secret_access_key"],
        region_name=credentials["region"],
    )

    try:
        bucket_name = credentials["bucket_name"]
        s3.upload_file(file_path, bucket_name, s3_key)
        logging.info(
            f"File {file_path} uploaded to S3 bucket {bucket_name}. Key: {s3_key}"
        )
    except Exception as e:
        logging.error(f"Error uploading file to S3: {e}")


def main():
    parser = argparse.ArgumentParser(description="MCMC Simulation for Price Prediction")
    parser.add_argument("-f", "--file", required=True, help="Path to JSON file")
    parser.add_argument("-c", "--asset_id", required=True, help="Asset/Ticker/Coin ID")
    parser.add_argument(
        "-p",
        "--start_price",
        default="latest",
        help="Start price (optional, default is latest price)",
    )
    parser.add_argument("-steps", type=int, default=7, help="Number of steps")
    parser.add_argument("-runs", type=int, default=100, help="Number of runs")
    parser.add_argument(
        "-sims", type=int, default=1000, help="Number of simulations per run"
    )
    parser.add_argument(
        "-shock_steps",
        type=str,
        default="1",
        help="Comma-separated string of shock steps",
    )
    parser.add_argument(
        "-shock_factors",
        type=str,
        default="1",
        help="Comma-separated string of shock factors",
    )

    args = parser.parse_args()

    json_file_path = args.file
    asset_id = args.asset_id
    start_price_arg = args.start_price
    steps = args.steps
    num_runs = args.runs
    num_simulations_per_run = args.sims
    shock_steps = list(map(int, args.shock_steps.split(",")))
    shock_factors = list(map(float, args.shock_factors.split(",")))
    max_workers = 50
    config_file = "../data/src/config.json"

    if len(shock_steps) != len(shock_factors):
        logging.error("Error: shock_steps and shock_factors must have the same length.")
        sys.exit(1)

    time_unit = determine_time_unit(json_file_path)
    logging.info(f"Time unit: {time_unit}")
    logging.info(f"Steps to simulate: {steps}")

    logging.info(f"Loading prices from {json_file_path}")
    prices = load_prices_from_json(json_file_path, asset_id)

    if start_price_arg.lower() == "latest":
        start_price = prices[-1]
    else:
        start_price = float(start_price_arg)

    logging.info(f"Applying a natural logarithm transformation to each price")
    log_prices = log_transform_prices(prices)

    logging.info(f"Creating a transition matrix from the log prices")
    transition_matrix = create_transition_matrix(log_prices)
    log_price_range = sorted(set(log_prices))

    logging.info(f"Normalizing the transition matrix")
    transition_matrix = normalize_transition_matrix(transition_matrix, log_price_range)

    log_start_price = np.log(start_price)

    logging.info(
        f"Running {num_runs} simulations with {num_simulations_per_run} paths each, including stress testing"
    )
    aggregated_path_counter, all_simulations, top_10_paths = aggregate_simulations(
        num_runs,
        transition_matrix,
        log_price_range,
        log_start_price,
        steps,
        num_simulations=num_simulations_per_run,
        max_workers=max_workers,
        shock_steps=shock_steps,
        shock_factors=shock_factors,
    )

    logging.info(
        f"Applying the exponential function to each log-transformed price to revert it to the original scale"
    )

    top_10_paths = [exp_transform_prices(path) for path in top_10_paths]

    logging.info(f"Visualizing the aggregated simulations")
    visualize_aggregated_simulations(
        top_10_paths,
        steps,
        asset_id,
        num_runs,
        num_simulations_per_run,
        time_unit,
        config_file,
    )


if __name__ == "__main__":
    main()

import argparse
import json
import logging
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pywt

from common import COIN_DATA_INPUT_FILE_DAILY

# COIN_DATA_INPUT_FILE = "../data/src/gme.json"

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def calculate_entropy(signal):
    """
    Calculate the Shannon entropy of a signal.

    :param signal: The signal to calculate entropy for, typically a wavelet transform level in this case

    :return: The Shannon entropy of the signal
    """
    histogram = np.histogram(signal, bins=50, density=True)[0]
    histogram = histogram[histogram > 0]  # Filter out zero values to avoid log(0)
    probability = histogram / histogram.sum()  # Normalize to make it a probability mass
    entropy = -np.sum(
        probability * np.log2(probability)
    )  # Use log base 2 for entropy in bits
    return entropy


def load_coin_data(filename, coin_id):
    """
    Load the price data for a specific coin from a JSON file.

    :param filename: The path to the JSON file containing the price data
    :param coin_id: The Coingecko ID of the coin to load data for

    :return: A tuple containing the prices and timestamps of the coin data
    """
    with open(filename, "r") as file:
        data = json.load(file)
        prices = [item[1] for item in data["data"][coin_id]["prices"]]
        timestamps = [item[0] for item in data["data"][coin_id]["prices"]]
    return prices, timestamps


def perform_wavelet_transform_and_calculate_entropy(prices):
    """
    Perform a wavelet transform on the price data and calculate the entropy of each level.

    :param prices: The price data to analyze

    :return: A tuple containing the wavelet coefficients and the entropy calculated for each level
    """
    logging.info("Performing wavelet transform and calculating entropy...")

    coeffs = pywt.wavedec(prices, "db1", level=4)
    entropy_per_level = [calculate_entropy(level) for level in coeffs]

    return coeffs, entropy_per_level


def process_coin_data(coin_id, input_filename):
    """
    Process the price data for a specific coin by performing a wavelet transform and calculating entropy.

    :param coin_id: The Coingecko ID of the coin to process
    :param input_filename: The path to the JSON file containing the price data

    :return: A tuple containing the coin ID and the entropy results for each wavelet level
    """
    logging.info(f"Loading and processing data for {coin_id}...")

    prices, _ = load_coin_data(input_filename, coin_id)
    coeffs, entropy_results = perform_wavelet_transform_and_calculate_entropy(prices)

    return coin_id, entropy_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze wavelet transform and entropy of cryptocurrency price data."
    )
    parser.add_argument("-c", "--base-coin", help="Coin ID to process")
    parser.add_argument(
        "--all", action="store_true", help="Process all coins in the dataset"
    )
    args = parser.parse_args()

    return args


def save_to_json(filename, timestamps, coeffs, entropy_results):
    """
    Save the wavelet coefficients and entropy results to a JSON file.

    :param filename: The path to the output JSON file
    :param timestamps: The timestamps corresponding to the price data
    :param coeffs: The wavelet coefficients to save
    :param entropy_results: The entropy results to save

    :return: None
    """
    data = {
        "timestamps": timestamps,
        "coefficients": [coeff.tolist() for coeff in coeffs],
        "entropy": entropy_results,  # Add entropy results to the JSON data
    }
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


def main():
    args = parse_args()

    if args.all:
        with open(COIN_DATA_INPUT_FILE_DAILY, "r") as file:
            data = json.load(file)

        all_entropy_results = {}
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(process_coin_data, coin_id, COIN_DATA_INPUT_FILE_DAILY)
                for coin_id in data["data"]
            ]
            for future in futures:
                coin_id, entropy_results = future.result()
                all_entropy_results[coin_id] = entropy_results

        # Save all entropy results (excluding wavelet coefficients) to a new JSON file to be visualized in the global
        # MOAC dashboard
        with open("../data/generated/all-wavelet-entropy.json", "w") as f:
            json.dump(all_entropy_results, f, indent=4)

        logging.info("All coins processed and entropy data saved.")

    elif args.base_coin:
        # Process a single coin to be visualized in the individual coin MOAC dashboard
        logging.info(
            f"Loading price data for {args.base_coin} from {COIN_DATA_INPUT_FILE_DAILY}."
        )
        prices, _ = load_coin_data(COIN_DATA_INPUT_FILE_DAILY, args.base_coin)
        coeffs, entropy_results = perform_wavelet_transform_and_calculate_entropy(
            prices
        )

        # For an individual coin, save both wavelet coefficients AND entropy
        output_filename = f"../data/generated/{args.base_coin}-wavelet-transform.json"

        with open(output_filename, "w") as f:
            json.dump(
                {
                    "coefficients": [coeff.tolist() for coeff in coeffs],
                    "entropy": entropy_results,
                },
                f,
                indent=4,
            )

        logging.info(f"Data for {args.base_coin} processed and saved.")
    else:
        raise ValueError(
            "No coin specified and '--all' flag not set. Please specify a coin ID or use the '--all' flag."
        )


if __name__ == "__main__":
    main()

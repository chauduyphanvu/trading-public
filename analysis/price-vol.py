import argparse
import json
import logging
import os

from common import COIN_DATA_INPUT_FILE_DAILY


# COIN_DATA_INPUT_FILE_DAILY = "../data/src/gme-close-2004-01-01.json"


def get_coin_data_input(file_name, coin_name, data_key):
    try:
        with open(file_name, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        logging.error(f"Error: The input file {file_name} does not exist.")
        return []
    except json.JSONDecodeError:
        logging.error(f"Error: The input file {file_name} is not valid JSON.")
        return []

    coin_data = data["data"].get(coin_name, {})
    return coin_data.get(data_key, [])


def save_data(coin_names, data_key, file_name_pattern, json_file_path):
    for coin_name in coin_names:
        data = get_coin_data_input(json_file_path, coin_name, data_key)
        file_name = file_name_pattern.format(coin_name)

        if os.path.exists(file_name):
            logging.info(f"Removing existing file: {file_name}")
            os.remove(file_name)

        if data:
            with open(file_name, "w") as f:
                json.dump(data, f)
            logging.info(
                f"{data_key.replace('_', ' ').capitalize()} data saved to {file_name}"
            )
        else:
            logging.error(
                f"No data found for '{data_key}' in the response for {coin_name}."
            )


def main():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Download and save coin data.")
    parser.add_argument(
        "--base-coin", nargs="+", help="List of coin IDs", required=True
    )
    args = parser.parse_args()
    coin_ids = args.base_coin

    save_data(
        coin_ids,
        "total_volumes",
        "../data/generated/{}-volume-data.json",
        COIN_DATA_INPUT_FILE_DAILY,
    )
    save_data(
        coin_ids,
        "prices",
        "../data/generated/{}-price-data.json",
        COIN_DATA_INPUT_FILE_DAILY,
    )
    save_data(
        coin_ids,
        "market_caps",
        "../data/generated/{}-market-cap-data.json",
        COIN_DATA_INPUT_FILE_DAILY,
    )


if __name__ == "__main__":
    main()

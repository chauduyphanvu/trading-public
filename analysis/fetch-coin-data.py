import argparse
import concurrent.futures
import datetime
import json
import logging
import os
from datetime import datetime
from time import time

import firebase_admin
import pandas as pd
import requests
from firebase_admin import credentials
from firebase_admin import db

from common import COIN_DATA_INPUT_FILE_DAILY, COIN_DATA_INPUT_FILE_HOURLY

COIN_IDS_FILE_DAILY = "../data/src/coingecko-coin-ids-daily.json"
COIN_IDS_FILE_HOURLY = "../data/src/coingecko-coin-ids-hourly.json"
OUTPUT_FILE_DAILY = COIN_DATA_INPUT_FILE_DAILY
OUTPUT_FILE_HOURLY = COIN_DATA_INPUT_FILE_HOURLY
LOG_TAG = "[COIN_DATA_FETCHER]"
CONFIG_FILE = "../data/src/config.json"
CRED_PATH = "../trading-b71a0-firebase-adminsdk-f7lq1-63ae25231d.json"
FIREBASE_DATABASE_URL = "https://trading-b71a0-default-rtdb.firebaseio.com/"

PARAMS_DAILY = {"vs_currency": "usd", "days": "max", "interval": "daily"}
PARAMS_HOURLY = {
    "vs_currency": "usd",
    "days": "90",
}  # For hourly, explicitly leave out `interval` because Coingecko API will know what to do.


def init_firebase():
    cred_path = "../trading-b71a0-firebase-adminsdk-f7lq1-63ae25231d.json"
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred, {"databaseURL": FIREBASE_DATABASE_URL})


def push_data_to_firebase(data):
    logging.info(f"{LOG_TAG} Pushing data to Firebase...")

    # Clear database before pushing to ensure idempotency
    clear_database("/coin_data")

    ref = db.reference("/coin_data")
    ref.set(data)

    logging.info(f"{LOG_TAG} Data pushed to Firebase at {pd.Timestamp.now()}.")


def clear_database(path):
    db.reference(path).delete()


def should_fetch(output_file, force_fetch):
    if force_fetch:
        logging.info(
            f"{LOG_TAG} Force fetching data regardless of time since last fetch."
        )
        return True

    if os.path.exists(output_file):
        with open(output_file, "r") as file:
            try:
                data = json.load(file)
                last_fetch_time = data.get("fetched_at")
                if last_fetch_time and (time() - last_fetch_time < 4 * 3600):
                    next_allowed_time = datetime.fromtimestamp(
                        last_fetch_time + 4 * 3600
                    )
                    logging.info(
                        f"{LOG_TAG} Data was fetched at {datetime.fromtimestamp(last_fetch_time).strftime('%Y-%m-%d %H:%M:%S')}. "
                        f"Try again at {next_allowed_time.strftime('%Y-%m-%d %H:%M:%S')}."
                    )
                    return False
            except (IOError, json.JSONDecodeError):
                pass  # File exists but is corrupt or empty, proceed to fetch
    return True


def fetch_prices_and_volumes(coin_name, params, api_key):
    logging.info(f"{LOG_TAG} {coin_name}: Fetching data from CoinGecko API...")

    url = f"https://pro-api.coingecko.com/api/v3/coins/{coin_name}/market_chart"
    params["x_cg_pro_api_key"] = api_key

    try:
        response = requests.get(url, params=params)
        data = response.json()

        if "prices" not in data or "total_volumes" not in data:
            logging.info(
                f"{LOG_TAG} No data available for {coin_name}. Check if the coin name, API key, and the params are "
                f"correct."
            )
            return None

        logging.info(f"{LOG_TAG} {coin_name}: Data fetched successfully.")
        return data
    except requests.RequestException as e:
        logging.error(f"{LOG_TAG} Failed to fetch data for {coin_name}: {e}")
    return None


def write_json_file(file_path, data):
    logging.info(f"{LOG_TAG} Writing coin data to {file_path}...")

    # Delete the file if it exists to ensure idempotency
    if os.path.exists(file_path):
        os.remove(file_path)

    try:
        with open(file_path, "w") as file:
            json.dump(data, file, indent=4)
    except Exception as e:
        logging.error(f"{LOG_TAG} An error occurred while writing the file: {e}")
        return False

    logging.info(
        f"{LOG_TAG} Coin data written to JSON file successfully at {pd.Timestamp.now()}."
    )
    return True


def fetch_all_data(coins, params, api_key):
    num_workers = min(100, len(coins), os.cpu_count() * 2)
    logging.info(
        f"{LOG_TAG} Fetching data concurrently for coins: {coins} with {num_workers} workers..."
    )

    all_data = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_coin = {
            executor.submit(fetch_prices_and_volumes, coin, params, api_key): coin
            for coin in coins
        }
        for future in concurrent.futures.as_completed(future_to_coin):
            coin = future_to_coin[future]
            try:
                data = future.result()
                if data:
                    all_data[coin] = data
            except Exception as exc:
                logging.info(f"{LOG_TAG} {coin} generated an exception: {exc}")
    return all_data


def read_coin_ids(file_path):
    logging.info(f"{LOG_TAG} Reading coin IDs from {file_path}...")

    try:
        with open(file_path, "r") as file:
            coin_data = json.load(file)
            return coin_data["ids"]
    except Exception as e:
        logging.info(f"{LOG_TAG} Failed to read coin IDs from {file_path}: {e}")
        return []


def main(api_key, coins, params, output_file, force_fetch=False):
    if should_fetch(output_file, force_fetch):
        data = fetch_all_data(coins, params, api_key)
        payload = {
            "fetched_at": int(time()),
            "endpoint": "market_chart",
            "days": params["days"],
            "interval": "daily" if params == PARAMS_DAILY else "hourly",
            "coin_ids": coins,
            "data": data,
        }

        if data:
            write_json_file(output_file, payload)

        # if data and write_json_file(output_file, payload):
        #     push_data_to_firebase(payload)
    else:
        logging.info(f"{LOG_TAG} Fetch skipped as it's within a 4-hour window.")


def read_api_key(file_path):
    try:
        with open(file_path, "r") as file:
            config = json.load(file)
            return config["coingecko"]["api_key_analyst"]
    except Exception as e:
        logging.info(f"{LOG_TAG} Failed to read API key from {file_path}: {e}")
        return ""


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    parser = argparse.ArgumentParser(description="Fetch and update coin data.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force fetch data regardless of the last fetched time.",
    )
    parser.add_argument(
        "-i",
        "--interval",
        choices=["daily", "hourly"],
        default="daily",
        help="Set the interval for fetching data: 'daily' or 'hourly'.",
    )
    args = parser.parse_args()

    api_key = read_api_key(CONFIG_FILE)

    if args.interval == "hourly":
        coin_ids_file = COIN_IDS_FILE_HOURLY
        params = PARAMS_HOURLY
        output_file = OUTPUT_FILE_HOURLY
    else:
        coin_ids_file = COIN_IDS_FILE_DAILY
        params = PARAMS_DAILY
        output_file = OUTPUT_FILE_DAILY

    coin_ids = read_coin_ids(coin_ids_file)

    init_firebase()
    main(api_key, coin_ids, params, output_file, args.force)

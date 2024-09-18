"""
Fetch stock and crypto aggregates (i.e., OHLCV data) from Polygon.io for a given date range and timespan and save the
data to a JSON file. If no ticker is provided, the script will read tickers from a local JSON file and fetch data for
each ticker concurrently.

Usage: python3 polygon-fetch.py -c <ticker> --start-date <YYYY-MM-DD> --timespan <timespan> --multiplier <multiplier>

Arguments:
    -c, --ticker       The ticker symbol of the stock or crypto (optional)
    --start-date       The start date in 'YYYY-MM-DD' format
    --end-date         The end date in 'YYYY-MM-DD' format (default: today's date)
    --timespan         The timespan (minute, hour, day, etc.)
    --multiplier       The multiplier for the timespan  # 15, 30, 60, etc.

Example: python3 polygon-fetch.py -c gme --start-date 2000-06-11 --multiplier 15 --timespan minute
"""

import argparse
import glob
import json
import logging
import os
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from polygon import RESTClient

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - [Polygon.io] %(message)s"
)

API_KEY = ""
client = RESTClient(API_KEY)


# Function to fetch aggregates for a single ticker
def fetch_aggregates(ticker, start_date, end_date, timespan, multiplier):
    logging.info(
        f"Fetching aggregates for ${ticker} from {start_date} to {end_date} with a multiplier of {multiplier}..."
    )

    try:
        aggs = []
        for a in client.list_aggs(
            ticker=ticker,
            multiplier=multiplier,
            timespan=timespan,
            from_=start_date,
            to=end_date,
            limit=50000,
        ):
            aggs.append(a.__dict__)
    except Exception as e:
        logging.error(f"Error fetching data: {e}")
        return

    if not aggs:
        logging.warning(f"No data found for ${ticker} in the given date range.")
        return

    actual_start_timestamp = aggs[0]["timestamp"]
    actual_end_timestamp = aggs[-1]["timestamp"]

    actual_start_date = time.strftime(
        "%Y-%m-%d", time.localtime(actual_start_timestamp / 1000)
    )
    actual_end_date = time.strftime(
        "%Y-%m-%d", time.localtime(actual_end_timestamp / 1000)
    )

    logging.info(
        f"Data fetched for ${ticker} from {actual_start_date} to {actual_end_date}."
    )

    # Format the data
    if ticker.startswith("X:"):
        ticker = ticker[2:]  # Drop the `x:` prefix
        if ticker.endswith("USD"):
            ticker = ticker[:-3]  # Drop the `usd` suffix

    formatted_data = {
        "fetched_at": int(time.time()),
        "endpoint": "aggregates",
        "days": f"{actual_start_date}-to-{actual_end_date}",
        "interval": f"{multiplier}-{timespan}",
        "coin_ids": [ticker.lower()],
        "data": {
            ticker.lower(): {
                "prices": [[agg["timestamp"], agg["close"]] for agg in aggs],
                "volumes": [[agg["timestamp"], agg["volume"]] for agg in aggs],
            }
        },
    }

    aggs_json = json.dumps(formatted_data, indent=4)

    directory = "../data/src/"
    if not os.path.exists(directory):
        os.makedirs(directory)

    for file in glob.glob(os.path.join(directory, f"{ticker.lower()}-*.json")):
        os.remove(file)
        logging.info(f"Deleted {file}")

    filename = os.path.join(
        directory,
        f"{ticker.lower()}-{multiplier}-{timespan}-{actual_start_date}-to-{actual_end_date}.json",
    )
    try:
        with open(filename, "w") as f:
            f.write(aggs_json)
        logging.info(f"Data saved to {filename}")
    except IOError as e:
        logging.error(f"Error saving data: {e}")


# Function to read tickers from a local JSON file
def read_tickers_from_file(file_path="../data/src/stock-tickers.json"):
    try:
        with open(file_path, "r") as f:
            tickers = json.load(f)
        return tickers
    except FileNotFoundError:
        logging.error(f"Ticker file not found: {file_path}")
        return []
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from file: {file_path}")
        return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch stock/crypto aggregates from Polygon.io"
    )
    parser.add_argument(
        "-c", "--ticker", type=str, help="The ticker symbol of the stock"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="The start date in 'YYYY-MM-DD' format",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=datetime.today().strftime("%Y-%m-%d"),
        help="The end date in 'YYYY-MM-DD' format (default: today's date)",
    )
    parser.add_argument(
        "--timespan",
        type=str,
        required=True,
        help="The timespan (minute, hour, day, etc.)",
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        required=True,
        help="The multiplier for the timespan",
    )

    args = parser.parse_args()

    if args.ticker:
        # Fetch data for the provided ticker
        fetch_aggregates(
            args.ticker.upper(),
            args.start_date,
            args.end_date,
            args.timespan,
            args.multiplier,
        )
    else:
        # Fetch data concurrently for tickers listed in the JSON file
        tickers = read_tickers_from_file()
        if tickers:
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        fetch_aggregates,
                        ticker.upper(),
                        args.start_date,
                        args.end_date,
                        args.timespan,
                        args.multiplier,
                    )
                    for ticker in tickers
                ]
                for future in futures:
                    future.result()  # Wait for all threads to complete

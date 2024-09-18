import argparse
import json
import logging
import os
from collections import defaultdict
from datetime import datetime

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

crypto_data = {
    "bitcoin": {"threshold": 10, "is_crypto": True},
    "shiba-inu": {"threshold": 5, "is_crypto": True},
    "myro": {"threshold": 10, "is_crypto": True},
    "dogecoin": {"threshold": 20, "is_crypto": True},
    "solana": {"threshold": 10, "is_crypto": True},
    "dogwifcoin": {"threshold": 10, "is_crypto": True},
    "gme": {"threshold": 10, "is_crypto": True},
    "aapl": {"threshold": 10, "is_crypto": False},
    "amzn": {"threshold": 10, "is_crypto": False},
    "bonk": {"threshold": 5, "is_crypto": True},
    "musa": {"threshold": 10, "is_crypto": False},
    "sbux": {"threshold": 10, "is_crypto": False},
    "tsla": {"threshold": 10, "is_crypto": False},
}


def filter_data_by_start_date(data, start_date):
    start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
    return [entry for entry in data if entry[0] >= start_timestamp]


def detect_price_swings(data, threshold):
    swings = []
    for i in range(1, len(data)):
        timestamp_previous, price_previous = data[i - 1]
        timestamp_current, price_current = data[i]
        if price_previous == 0:
            continue
        percent_change = ((price_current - price_previous) / price_previous) * 100
        if abs(percent_change) >= threshold:
            swings.append(
                {
                    "from": timestamp_previous,
                    "to": timestamp_current,
                    "change": percent_change,
                }
            )
    return swings


def calculate_average_returns(data, is_crypto):
    daily_returns = defaultdict(list)
    day_counts = defaultdict(int)
    week_day_returns = defaultdict(list)

    positive_day_counts = defaultdict(int)
    total_day_counts = defaultdict(int)
    positive_week_day_counts = defaultdict(int)
    total_week_day_counts = defaultdict(int)

    for i in range(1, len(data)):
        # Important: Daily prices returned by Coingecko are collected at 19:00:00 EST (or 00:00:00 GMT) of each day.
        # That means that price difference / percent change calculations are between 19:00:00 EST on the current day and
        # 19:00:00 EST on the previous day.
        #
        # Implication: Even if a day is technically down from midnight to midnight, the price difference (and therefore
        # the percent change) could be positive if the price at 19:00:00 EST is higher than the price at 19:00:00 EST
        # the previous day.
        #
        # When checking the matrix, mentally assume that the probability and average return calculations are for data
        # since the previous day's 19:00:00 EST price.
        timestamp_previous, price_previous = data[i - 1]
        timestamp_current, price_current = data[i]
        if price_previous == 0:
            continue
        date_current = datetime.fromtimestamp(timestamp_current / 1000.0)
        day_of_week = date_current.strftime("%A")

        if not is_crypto and day_of_week in ["Saturday", "Sunday"]:
            continue

        percent_change = ((price_current - price_previous) / price_previous) * 100
        daily_returns[day_of_week].append(percent_change)
        day_counts[day_of_week] += 1

        week_of_month = (date_current.day - 1) // 7 + 1
        week_day_returns[(week_of_month, day_of_week)].append(percent_change)

        logging.info(f"{date_current}: {percent_change:.2f}%")

        # Day before --> current day has a positive return
        # "Current day" is NOT the same as "today" in the context of the data collection time
        if percent_change > 0:
            positive_day_counts[day_of_week] += 1
            positive_week_day_counts[(week_of_month, day_of_week)] += 1

        total_day_counts[day_of_week] += 1
        total_week_day_counts[(week_of_month, day_of_week)] += 1

    average_returns = {
        day: sum(returns) / len(returns) for day, returns in daily_returns.items()
    }
    average_week_day_returns = {
        key: sum(returns) / len(returns) for key, returns in week_day_returns.items()
    }

    day_probabilities = {
        day: positive_day_counts[day] / total_day_counts[day] for day in day_counts
    }
    week_day_probabilities = {
        key: positive_week_day_counts[key] / total_week_day_counts[key]
        for key in total_week_day_counts
    }

    return (
        daily_returns,
        average_returns,
        week_day_returns,
        average_week_day_returns,
        day_probabilities,
        week_day_probabilities,
    )


def convert_tuple_keys_to_str(d):
    return {str(k): v for k, v in d.items()}


def main():
    parser = argparse.ArgumentParser(
        description="Detect drastic price swings in cryptocurrency data and find the best day to buy."
    )

    parser.add_argument(
        "-c",
        "--base-coin",
        required=True,
        help='ID of the coin (e.g., "bitcoin", "ethereum"). Must be one of: '
        + ", ".join(crypto_data.keys()),
    )

    parser.add_argument(
        "-s",
        "--start-date",
        help='Start date for the analysis in the format "YYYY-MM-DD". If not specified, the whole dataset will be used.',
    )

    # Accept file name as `-f` command line argument
    parser.add_argument(
        "-f",
        "--file",
        required=True,
        help="Path to the JSON file containing the stock price data."
    )

    args = parser.parse_args()
    coin_info = crypto_data.get(args.base_coin)
    if not coin_info:
        print(
            f"Error: Invalid coin ID {args.base_coin}. Must be one of: "
            + ", ".join(crypto_data.keys())
        )
        exit(1)

    threshold = coin_info["threshold"]
    is_crypto = coin_info["is_crypto"]

    # Use the file argument for reading the price data
    price_data_filename = args.file
    swings_data_filename = f"../data/generated/{args.base_coin}-price-swings.json"

    try:
        with open(price_data_filename, "r") as file:
            data = json.load(file)
            # Extract the relevant price data based on the coin ID
            if args.base_coin not in data["data"]:
                print(f"Error: No data available for {args.base_coin} in the file.")
                exit(1)

            # Extract prices list from the nested JSON structure
            data = data["data"][args.base_coin]["prices"]
    except FileNotFoundError:
        print(f"Error: File {price_data_filename} not found.")
        exit(1)
    except json.JSONDecodeError:
        print(f"Error: File {price_data_filename} is not a valid JSON.")
        exit(1)

    if args.start_date:
        logging.info(f"Start date provided: {args.start_date}. Filtering data...")
        data = filter_data_by_start_date(data, args.start_date)

    price_swings = detect_price_swings(data, threshold)
    (
        daily_returns,
        average_returns,
        week_day_returns,
        average_week_day_returns,
        day_probabilities,
        week_day_probabilities,
    ) = calculate_average_returns(data, is_crypto)

    best_days = sorted(average_returns, key=average_returns.get)
    logging.info(f"Best days to buy: {best_days}")

    best_week_days = sorted(average_week_day_returns, key=average_week_day_returns.get)
    logging.info(f"Best week days to buy: {best_week_days}")

    week_day_returns_str = convert_tuple_keys_to_str(week_day_returns)
    average_week_day_returns_str = convert_tuple_keys_to_str(average_week_day_returns)
    week_day_probabilities_str = convert_tuple_keys_to_str(week_day_probabilities)

    if os.path.exists(swings_data_filename):
        logging.info(f"Removing existing file {swings_data_filename}")
        os.remove(swings_data_filename)

    with open(swings_data_filename, "w") as file:
        json_output = {
            "swings": price_swings,
            "threshold": threshold,
            "average_returns": average_returns,
            "daily_returns": daily_returns,
            "week_day_returns": week_day_returns_str,
            "average_week_day_returns": average_week_day_returns_str,
            "best_days": best_days,
            "best_week_days": best_week_days,
            "day_probabilities": day_probabilities,
            "week_day_probabilities": week_day_probabilities_str,
        }
        json.dump(json_output, file, indent=4)
        logging.info(f"Data saved to {swings_data_filename}")


if __name__ == "__main__":
    main()

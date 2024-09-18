import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp
import pytz
import seaborn as sns

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def read_json_file(file_path):
    logging.info(f"Reading JSON data from file: {file_path}")
    with open(file_path, "r") as file:
        data = json.load(file)
    logging.info(f"Successfully read JSON data from file: {file_path}")
    return data


def maximize_daily_profit(day_prices):
    logging.info(f"Setting up ILP model for day prices: {day_prices}")
    model = pulp.LpProblem("Maximize_Profit", pulp.LpMaximize)

    n = len(day_prices)
    buy = pulp.LpVariable.dicts("Buy", range(n), cat=pulp.LpBinary)
    sell = pulp.LpVariable.dicts("Sell", range(n), cat=pulp.LpBinary)

    profit = pulp.lpSum(
        sell[i] * day_prices[i][1] - buy[i] * day_prices[i][1] for i in range(n)
    )
    model += profit
    logging.info(f"Objective function set up to maximize profit: {profit}")

    model += pulp.lpSum(buy[i] for i in range(n)) == 1, "One Buy"
    model += pulp.lpSum(sell[i] for i in range(n)) == 1, "One Sell"
    for i in range(n):
        for j in range(i + 1, n):
            model += buy[j] <= sell[i], f"Buy_after_sell_{i}_{j}"

    logging.info("Constraints added to the model")

    logging.info("Solving the ILP model")
    model.solve()
    logging.info("Model solved")

    buy_time = next(i for i in range(n) if pulp.value(buy[i]) == 1)
    sell_time = next(i for i in range(n) if pulp.value(sell[i]) == 1)
    logging.info(
        f"Optimal buy time: {buy_time}, sell time: {sell_time}, profit: {pulp.value(model.objective)}"
    )

    return (
        day_prices[buy_time][0],
        day_prices[sell_time][0],
        pulp.value(model.objective),
    )


def average_time(times):
    avg_seconds = np.mean([t.hour * 3600 + t.minute * 60 + t.second for t in times])
    avg_hour = int(avg_seconds // 3600)
    avg_seconds %= 3600
    avg_minute = int(avg_seconds // 60)
    avg_second = int(avg_seconds % 60)
    average_dt = datetime.combine(datetime.today(), datetime.min.time()) + timedelta(
        hours=avg_hour, minutes=avg_minute, seconds=avg_second
    )
    logging.info(f"Average time calculated: {average_dt}")
    return average_dt


def find_lowest_price_hour(day_prices):
    hourly_prices = defaultdict(list)
    for timestamp, price in day_prices:
        dt = datetime.fromtimestamp(timestamp / 1000, tz=pytz.utc)
        hour = dt.hour
        hourly_prices[hour].append(price)
    avg_hourly_prices = {
        hour: np.mean(prices) for hour, prices in hourly_prices.items()
    }
    lowest_price_hour = min(avg_hourly_prices, key=avg_hourly_prices.get)
    return lowest_price_hour


def process_prices(prices):
    prices_by_day_and_week = defaultdict(lambda: defaultdict(list))
    for timestamp, price in prices:
        dt = datetime.fromtimestamp(timestamp / 1000, tz=pytz.utc)
        day_of_week = dt.weekday()
        week_of_month = (dt.day - 1) // 7 + 1
        prices_by_day_and_week[day_of_week][week_of_month].append((timestamp, price))
    logging.info(
        f"Grouped prices by day of the week and week of the month: {prices_by_day_and_week}"
    )
    return prices_by_day_and_week


def gather_results(prices_by_day_and_week):
    desired_timezone = pytz.timezone("US/Eastern")
    all_results = defaultdict(
        lambda: defaultdict(lambda: {"buy_times": [], "sell_times": []})
    )
    for day, weeks in prices_by_day_and_week.items():
        for week, day_prices in weeks.items():
            logging.info(f"Processing day: {day}, week: {week}")
            daily_prices_by_date = defaultdict(list)
            for timestamp, price in day_prices:
                date = datetime.fromtimestamp(timestamp / 1000, tz=pytz.utc).date()
                daily_prices_by_date[date].append((timestamp, price))

            for date, prices in daily_prices_by_date.items():
                if prices:
                    lowest_price_hour = find_lowest_price_hour(prices)
                    filtered_prices = [
                        (timestamp, price)
                        for timestamp, price in prices
                        if datetime.fromtimestamp(timestamp / 1000, tz=pytz.utc).hour
                        >= lowest_price_hour
                    ]
                    if filtered_prices:
                        buy_time, sell_time, profit = maximize_daily_profit(
                            filtered_prices
                        )
                        all_results[day][week]["buy_times"].append(
                            datetime.fromtimestamp(buy_time / 1000, tz=pytz.utc)
                        )
                        all_results[day][week]["sell_times"].append(
                            datetime.fromtimestamp(sell_time / 1000, tz=pytz.utc)
                        )
                        logging.info(
                            f"Processed date: {date}, buy time: {buy_time}, sell time: {sell_time}, profit: {profit}"
                        )
    return all_results, desired_timezone


def calculate_avg_times(all_results, desired_timezone):
    avg_results = defaultdict(lambda: defaultdict(dict))
    for day, weeks in all_results.items():
        for week, times in weeks.items():
            if times["buy_times"] and times["sell_times"]:
                logging.info(
                    f"Calculating average buy and sell times for day: {day}, week: {week}"
                )
                avg_buy_time = average_time(times["buy_times"])
                avg_sell_time = average_time(times["sell_times"])
                avg_results[day][week] = {
                    "avg_buy_time": avg_buy_time.astimezone(desired_timezone),
                    "avg_sell_time": avg_sell_time.astimezone(desired_timezone),
                }
                logging.info(
                    f"Calculated average buy and sell times for day: {day}, week: {week}"
                )
    return avg_results


def plot_results(avg_results, coin_name):
    day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    def time_to_float(t):
        return t.hour + t.minute / 60 + t.second / 3600

    buy_times_data = pd.DataFrame(index=day_names, columns=range(1, 6))
    sell_times_data = pd.DataFrame(index=day_names, columns=range(1, 6))

    for day, weeks in avg_results.items():
        for week, result in weeks.items():
            buy_times_data.at[day_names[day], week] = time_to_float(
                result["avg_buy_time"]
            )
            sell_times_data.at[day_names[day], week] = time_to_float(
                result["avg_sell_time"]
            )

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    sns.heatmap(
        data=buy_times_data.astype(float),
        annot=True,
        fmt=".2f",
        cmap="Reds",
        cbar=True,
        linewidths=0.5,
    )
    plt.title(f"{coin_name.upper()} - Average Entry (hours since midnight)")
    plt.xlabel("Week of the Month")
    plt.ylabel("Day of the Week")

    plt.subplot(1, 2, 2)
    sns.heatmap(
        data=sell_times_data.astype(float),
        annot=True,
        fmt=".2f",
        cmap="Greens",
        cbar=True,
        linewidths=0.5,
    )
    plt.title(f"{coin_name.upper()} - Average Exit (hours since midnight)")
    plt.xlabel("Week of the Month")
    plt.ylabel("Day of the Week")

    plt.suptitle(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    plt.show()


def print_schedule(avg_results):
    day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    print("\nOptimal Trading Schedule:")
    for day, weeks in avg_results.items():
        for week, times in weeks.items():
            buy_time = times["avg_buy_time"].strftime("%H:%M:%S")
            sell_time = times["avg_sell_time"].strftime("%H:%M:%S")
            print(
                f"{day_names[day]} (Week {week}): Buy at {buy_time}, Sell at {sell_time}"
            )


def main():
    parser = argparse.ArgumentParser(description="Optimize crypto trading.")
    parser.add_argument("-f", "--file", required=True, help="Path to the JSON file")
    parser.add_argument("-c", "--coin", required=True, help="ID of the coin to analyze")
    parser.add_argument(
        "-b",
        "--balance",
        type=float,
        default=80000,
        help="Initial balance for backtesting",
    )
    args = parser.parse_args()
    logging.info(f"Parsed command line arguments: {args}")

    crypto_data = read_json_file(args.file)

    coin_name = args.coin
    logging.info(f"Extracting price data for coin: {coin_name}")
    if (coin_data := crypto_data["data"].get(coin_name)) is None:
        logging.error(f"Coin {coin_name} not found in the data.")
        exit(1)

    prices = coin_data["prices"]
    logging.info(f"Extracted price data for coin: {coin_name}")

    prices_by_day_and_week = process_prices(prices)
    all_results, desired_timezone = gather_results(prices_by_day_and_week)
    avg_results = calculate_avg_times(all_results, desired_timezone)
    plot_results(avg_results, coin_name)


if __name__ == "__main__":
    main()

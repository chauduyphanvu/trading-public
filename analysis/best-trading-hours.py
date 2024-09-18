import argparse
import json
import logging
from datetime import datetime, timedelta

import numpy as np  # Import numpy for averaging times
import pulp
import pytz

from common import COIN_DATA_INPUT_FILE_HOURLY

# python3 best-trading-hours.py -c dogwifcoin | grep "INFO"

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Function to read JSON data from file
def read_json_file(file_path):
    logging.info(f"Reading JSON data from file: {file_path}")
    with open(file_path, "r") as file:
        data = json.load(file)
    logging.info(f"Successfully read JSON data from file: {file_path}")
    return data


# Define the ILP model
def maximize_daily_profit(day_prices):
    logging.info(f"Setting up ILP model for day prices: {day_prices}")
    model = pulp.LpProblem("Maximize_Profit", pulp.LpMaximize)

    # Define variables for buy and sell
    n = len(day_prices)
    buy = pulp.LpVariable.dicts("Buy", range(n), cat=pulp.LpBinary)
    sell = pulp.LpVariable.dicts("Sell", range(n), cat=pulp.LpBinary)

    # Objective function: Maximize profit
    profit = pulp.lpSum(
        sell[i] * day_prices[i][1] - buy[i] * day_prices[i][1] for i in range(n)
    )
    model += profit
    logging.info(f"Objective function set up to maximize profit: {profit}")

    # Constraints
    model += pulp.lpSum(buy[i] for i in range(n)) == 1, "One Buy"
    model += pulp.lpSum(sell[i] for i in range(n)) == 1, "One Sell"
    for i in range(n):
        for j in range(i + 1, n):
            model += buy[j] <= sell[i], f"Buy_after_sell_{i}_{j}"

    logging.info("Constraints added to the model")

    # Solve the model
    logging.info("Solving the ILP model")
    model.solve()
    logging.info("Model solved")

    # Extract results
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


# Function to calculate the average time
def average_time(times):
    logging.info(f"Calculating average time for times: {times}")
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


# Command line argument parsing
parser = argparse.ArgumentParser(description="Optimize crypto trading.")
parser.add_argument("-c", "--coin", required=True, help="ID of the coin to analyze")
args = parser.parse_args()
logging.info(f"Parsed command line arguments: {args}")

# Read JSON data from file
crypto_data = read_json_file(COIN_DATA_INPUT_FILE_HOURLY)

# Extract price data for the specified coin
coin_name = args.coin
logging.info(f"Extracting price data for coin: {coin_name}")
if (coin_data := crypto_data["data"].get(coin_name)) is None:
    logging.error(f"Coin {coin_name} not found in the data.")
    exit(1)

prices = coin_data["prices"]
logging.info(f"Extracted price data for coin: {coin_name}")

# Convert timestamps to datetime and group by specific days
prices_by_day = {i: [] for i in range(7)}
for timestamp, price in prices:
    dt = datetime.fromtimestamp(timestamp / 1000, tz=pytz.utc)
    day_of_week = dt.weekday()
    prices_by_day[day_of_week].append((timestamp, price))
logging.info(f"Grouped prices by day of the week: {prices_by_day}")

# Define the desired timezone (Eastern Time)
desired_timezone = pytz.timezone("US/Eastern")

# Apply the model to each day and gather all the buy and sell times
all_results = {i: {"buy_times": [], "sell_times": []} for i in range(7)}
for day, day_prices in prices_by_day.items():
    logging.info(f"Processing day: {day}")
    daily_prices_by_date = {}
    for timestamp, price in day_prices:
        date = datetime.fromtimestamp(timestamp / 1000, tz=pytz.utc).date()
        if date not in daily_prices_by_date:
            daily_prices_by_date[date] = []
        daily_prices_by_date[date].append((timestamp, price))

    for date, prices in daily_prices_by_date.items():
        if prices:
            buy_time, sell_time, profit = maximize_daily_profit(prices)
            all_results[day]["buy_times"].append(
                datetime.fromtimestamp(buy_time / 1000, tz=pytz.utc)
            )
            all_results[day]["sell_times"].append(
                datetime.fromtimestamp(sell_time / 1000, tz=pytz.utc)
            )
            logging.info(
                f"Processed date: {date}, buy time: {buy_time}, sell time: {sell_time}, profit: {profit}"
            )
            if day == 0:  # Logging specific to Monday
                logging.info(
                    f"Monday trade - Date: {date}, Buy time: {datetime.fromtimestamp(buy_time / 1000, tz=pytz.utc)}, Sell time: {datetime.fromtimestamp(sell_time / 1000, tz=pytz.utc)}, Profit: {profit}"
                )

# Calculate average buy and sell times
avg_results = {}
for day, times in all_results.items():
    if times["buy_times"] and times["sell_times"]:
        logging.info(f"Calculating average buy and sell times for day: {day}")
        avg_buy_time = average_time(times["buy_times"])
        avg_sell_time = average_time(times["sell_times"])
        avg_results[day] = {
            "avg_buy_time": avg_buy_time.astimezone(desired_timezone),
            "avg_sell_time": avg_sell_time.astimezone(desired_timezone),
        }
        logging.info(f"Calculated average buy and sell times for day: {day}")

# Map day indices to day names
day_names = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
]

# Output results with day names and 12-hour format times
for day, result in avg_results.items():
    avg_buy_time_12hr = result["avg_buy_time"].strftime("%I:%M:%S %p")
    avg_sell_time_12hr = result["avg_sell_time"].strftime("%I:%M:%S %p")
    logging.info(
        f"Day: {day_names[day]}, Average Buy at: {avg_buy_time_12hr} (EST), Average Sell at: {avg_sell_time_12hr} (EST)"
    )
    print(f"Day: {day_names[day]}")
    print(f"  Average Buy at: {avg_buy_time_12hr} (EST)")
    print(f"  Average Sell at: {avg_sell_time_12hr} (EST)")

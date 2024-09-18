import os
import json
import numpy as np
import pandas as pd
from datetime import datetime

# Directory containing the JSON files
directory = "../data/src/"  # Update this with the correct path to your JSON files

# Dictionary to store average daily returns for each stock
stock_returns = {}

# Step 1: Find the earliest timestamp among all files
earliest_timestamp = None

for filename in os.listdir(directory):
    if "1-day" in filename and filename.endswith(".json"):  # Look for daily data files
        filepath = os.path.join(directory, filename)

        try:
            # Load JSON data
            with open(filepath, "r") as file:
                data = json.load(file)

            # Extract stock prices
            stock_symbol = list(data["data"].keys())[0]
            prices = data["data"][stock_symbol]["prices"]

            if not prices:
                print(f"No prices found in file {filename}. Skipping...")
                continue

            # Extract timestamps and find the minimum timestamp in the current file
            timestamps = [price[0] for price in prices]
            min_timestamp = min(timestamps)

            # Update the earliest timestamp
            if earliest_timestamp is None or min_timestamp < earliest_timestamp:
                earliest_timestamp = min_timestamp
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Error processing file {filename}: {e}. Skipping...")
            continue

# Check if earliest_timestamp is None before proceeding
if earliest_timestamp is None:
    raise ValueError(
        "No valid timestamps found in any JSON file. Please check the files."
    )

# Convert the earliest timestamp to a human-readable date
start_date = datetime.utcfromtimestamp(earliest_timestamp / 1000).strftime(
    "%Y-%m-%d %H:%M:%S"
)

# Step 2: Calculate daily returns from the common start date
for filename in os.listdir(directory):
    if "1-day" in filename and filename.endswith(".json"):  # Look for daily data files
        filepath = os.path.join(directory, filename)

        try:
            # Load JSON data
            with open(filepath, "r") as file:
                data = json.load(file)

            # Extract stock symbol and prices
            stock_symbol = list(data["data"].keys())[0]
            prices = data["data"][stock_symbol]["prices"]

            # Convert price data into a numpy array and filter based on the earliest timestamp
            prices = np.array(prices)
            filtered_prices = prices[prices[:, 0] >= earliest_timestamp]
            stock_prices = filtered_prices[:, 1].astype(
                float
            )  # Prices are in the second column

            # Check if there are enough prices to calculate returns
            if len(stock_prices) < 2:
                print(
                    f"Not enough data to calculate returns for {stock_symbol}. Skipping..."
                )
                continue

            # Calculate daily returns
            daily_returns = np.diff(stock_prices) / stock_prices[:-1]

            # Store the average daily return
            average_return = np.mean(daily_returns)
            stock_returns[stock_symbol] = average_return
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"Error processing file {filename}: {e}. Skipping...")
            continue

# Check if there are any calculated returns
if not stock_returns:
    raise ValueError(
        "No valid stock returns were calculated. Please check the data files."
    )

# Convert the dictionary to a DataFrame for easier manipulation
returns_df = pd.DataFrame(
    list(stock_returns.items()), columns=["Stock", "Average Daily Return"]
)

# Sort the DataFrame by average daily return in descending order
returns_df = returns_df.sort_values(by="Average Daily Return", ascending=False)

# Add an index column starting from 1
returns_df.reset_index(drop=True, inplace=True)
returns_df.index += 1
returns_df.index.name = "Index"

# Convert DataFrame to markdown format and print
markdown_table = returns_df.to_markdown()
print(markdown_table)

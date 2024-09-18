import argparse
import json
import threading
from datetime import datetime

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from common import COIN_DATA_INPUT_FILE_DAILY


def write_json_file(file_path, data):
    print(f"Writing data to {file_path}...")
    try:
        with open(file_path, "w") as file:
            if isinstance(data, pd.DataFrame):
                data = data.to_dict(orient="records")
            json.dump(data, file, indent=4)
    except Exception as e:
        print(f"An error occurred while writing the file: {e}")
        return False
    print("Data written to JSON file successfully.")
    return True


def load_prices(coin_id):
    try:
        with open(COIN_DATA_INPUT_FILE_DAILY, "r") as f:
            all_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {COIN_DATA_INPUT_FILE_DAILY} does not exist.")
        return pd.DataFrame()
    except json.JSONDecodeError:
        print(f"Error: File {COIN_DATA_INPUT_FILE_DAILY} is not valid JSON.")
        return pd.DataFrame()

    data = all_data["data"].get(coin_id, {})

    if "prices" not in data:
        print(f"No price data available for {coin_id}.")
        return pd.DataFrame()

    price_df = pd.DataFrame(data["prices"], columns=["timestamp", f"price_{coin_id}"])

    price_df[f"snapped_at_{coin_id}"] = pd.to_datetime(price_df["timestamp"], unit="ms")

    price_df.set_index(f"snapped_at_{coin_id}", inplace=True)
    price_df.drop(columns=["timestamp"], inplace=True)

    return price_df


def load_all_data(coins, all_prices):
    for coin in coins:
        price_df = load_prices(coin)
        all_prices[coin] = price_df


def calculate_dynamic_correlations(all_data, base_coin, today):
    base_data = all_data[base_coin]
    results = {"base_coin": base_coin, "data": []}
    today = pd.Timestamp(today)  # Convert today's date for consistency

    # Starting from 1 day after the base coin's start date
    start_date = base_data.index.min() + pd.Timedelta(days=1)
    end_date = start_date

    while end_date <= today:
        day_correlations = []
        for coin, df_other in all_data.items():
            if coin == base_coin or df_other.empty:
                continue

            # Only consider data up to the current end_date if it exists
            if end_date <= df_other.index.max():
                df_base_filtered = base_data[
                    (base_data.index >= start_date) & (base_data.index <= end_date)
                ]
                df_other_filtered = df_other[
                    (df_other.index >= start_date) & (df_other.index <= end_date)
                ]

                if (
                    not df_base_filtered.empty
                    and not df_other_filtered.empty
                    and len(df_base_filtered) > 1
                    and len(df_other_filtered) > 1
                ):
                    correlation = df_base_filtered[f"price_{base_coin}"].corr(
                        df_other_filtered[f"price_{coin}"]
                    )
                    print(
                        f"Correlation between {base_coin} and {coin} on {end_date}: {correlation}"
                    )
                    if not pd.isna(correlation):
                        day_correlations.append(
                            {
                                "coin": coin,
                                "correlation": correlation,
                            }
                        )

        if day_correlations:
            results["data"].append(
                {
                    "end_date": end_date.strftime("%Y-%m-%d"),
                    "correlations": day_correlations,
                }
            )

        end_date += pd.Timedelta(days=1)  # Increment the end date by one day

    print(f"Results calculated for {base_coin}: {results}")
    return results


def threaded_load(coins):
    all_prices = {}
    threads = []
    chunk_size = 60
    coin_chunks = [coins[i : i + chunk_size] for i in range(0, len(coins), chunk_size)]
    for i, chunk in enumerate(coin_chunks):
        thread = threading.Thread(
            target=load_all_data,
            args=(chunk, all_prices),
        )
        threads.append(thread)
        thread.start()
    for thread in threads:
        thread.join()
    return all_prices


def calculate_corrs(base_coin, other_coins, file_path):
    all_prices = threaded_load([base_coin] + other_coins)
    today = datetime.now().date()
    result_data = calculate_dynamic_correlations(all_prices, base_coin, today)

    # Serialize and save price correlations data
    write_json_file(file_path, result_data)
    print(f"Correlation data saved to {file_path}")


def swap_base_coin_in_js(js_file_path, base_coin):
    try:
        with open(js_file_path, "r") as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"JS file '{js_file_path}' not found.")
        return False

    for i, line in enumerate(lines):
        if "const BASE_COIN" in line:
            lines[i] = f"const BASE_COIN = '{base_coin}';\n"
            break

    try:
        with open(js_file_path, "w") as file:
            file.writelines(lines)
    except Exception as e:
        print(f"An error occurred while modifying the JS file: {e}")
        return False

    print(f"Base coin value in {js_file_path} updated to '{base_coin}'.")
    return True


def create_correlation_chart(correlation_data, base_coin):
    # Accumulate data for plotting
    records = []
    for record in correlation_data["data"]:
        date = record["end_date"]
        for correlation in record["correlations"]:
            records.append(
                {
                    "date": date,
                    "coin": correlation["coin"],
                    "correlation": correlation["correlation"],
                }
            )

    # Create DataFrame from records
    correlation_df = pd.DataFrame(records)
    correlation_df["date"] = pd.to_datetime(correlation_df["date"])
    correlation_df.sort_values(by=["date", "coin"], inplace=True)

    # Create a line plot for correlations over time
    plt.figure(figsize=(14, 10))
    sns.lineplot(
        data=correlation_df, x="date", y="correlation", hue="coin", marker="o", alpha=1
    )

    plt.xlabel("Date")
    plt.ylabel("Correlation Coefficient")
    plt.title(f"{base_coin.upper()} | Price Correlation with Other Coins Over Time")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    plt.legend().set_visible(False)

    # Save the chart as an image file
    plt.tight_layout()
    plt.savefig(f"../data/generated/{base_coin}-correlation.png")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Fetch and calculate correlations for cryptocurrency data."
    )
    parser.add_argument(
        "--base-coin", required=True, help="The base coin for correlation comparison."
    )
    args = parser.parse_args()
    base_coin = args.base_coin

    with open("../data/src/coingecko-coin-ids-daily.json", "r") as file:
        coins_data = json.load(file)
    all_coins = coins_data["ids"]

    print(f"Calculating correlations for coin IDs: {all_coins}")

    other_coins = [coin for coin in all_coins if coin != base_coin]
    json_file_path = f"../data/generated/{base_coin}-price-corr.json"

    calculate_corrs(base_coin, other_coins, json_file_path)

    with open(json_file_path, "r") as f:
        correlation_data = json.load(f)

    create_correlation_chart(correlation_data, base_coin)

    js_file_path = "../dash/corrs.html"
    swap_base_coin_in_js(js_file_path, base_coin)


if __name__ == "__main__":
    main()

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from common import COIN_DATA_INPUT_FILE_DAILY


def load_coin_names(filename):
    with open(filename, "r") as file:
        coin_list = json.load(file)
        return coin_list["ids"]


def load_local_data(coin_id):
    try:
        with open(COIN_DATA_INPUT_FILE_DAILY, "r") as file:
            all_data = json.load(file)
    except FileNotFoundError:
        print(
            f"Error: The file {COIN_DATA_INPUT_FILE_DAILY} does not exist for {coin_id}"
        )
        return None
    except json.JSONDecodeError:
        print(
            f"Error: File {COIN_DATA_INPUT_FILE_DAILY} is not valid JSON for {coin_id}"
        )
        return None

    data = all_data["data"].get(coin_id, {})

    if "prices" not in data:
        print(f"No price data available for {coin_id}.")
        return None

    return data


def process_coin_data(coin_name):
    data = load_local_data(coin_name)
    if data:
        df_prices = pd.DataFrame(data["prices"], columns=["Timestamp", "Price"])
        df_prices["Timestamp"] = pd.to_datetime(df_prices["Timestamp"], unit="ms")
        df_prices.set_index("Timestamp", inplace=True)
        return df_prices
    return None


def align_datasets(data_frames, base_coin_start_date):
    """Align all dataframes to start from the base coin's start date."""
    aligned_data = {
        name: df[df.index >= base_coin_start_date]
        for name, df in data_frames.items()
        if df is not None
    }
    return aligned_data


def calculate_volatility(data_frames):
    volatility_data = {}
    for name, df in data_frames.items():
        df["Returns"] = df["Price"].pct_change()
        volatility = df["Returns"].std() * 100
        volatility_data[name] = {
            "volatility": volatility,
            "returns_data": [
                [int(ts.timestamp() * 1000), ret]
                for ts, ret in zip(df.index, df["Returns"])
                if not pd.isna(ret)
            ],
        }
    return volatility_data


def create_heatmap(data_frames, base_coin):
    # Combine returns data into a single DataFrame
    combined_returns = pd.DataFrame(
        {name: df["Returns"] for name, df in data_frames.items()}
    )

    # Handle extreme values by clipping
    combined_returns = combined_returns.clip(-1, 1)

    # Fill NaN values with zero
    combined_returns.fillna(0, inplace=True)

    # Check for any potential issues in the data
    print(combined_returns.index.unique())

    # Drop duplicated indices if any
    combined_returns = combined_returns[
        ~combined_returns.index.duplicated(keep="first")
    ]

    # Formatting the dates to be more readable
    combined_returns.index = pd.to_datetime(combined_returns.index).strftime("%Y-%m-%d")

    # Ensure unique dates after formatting
    combined_returns = combined_returns.loc[
        ~combined_returns.index.duplicated(keep="first")
    ]

    # Create the heatmap
    plt.figure(figsize=(15, 10))
    sns.heatmap(
        combined_returns.T,
        cmap="viridis",
        cbar=True,
        linewidths=0,
        linecolor="grey",
        robust=True,
    )
    plt.title(
        f"Crypto Returns Heatmap\nThe brighter the color, the higher the return (the higher the profit)"
    )
    plt.xlabel("Date")
    plt.ylabel("Coin")
    plt.xticks(rotation=45, ha="right")

    # Add explanation text
    plt.figtext(
        0.5,
        -0.1,
        "Intensity of colors: Darker colors (purple) represent negative returns, "
        "while lighter colors (yellow) represent positive returns. "
        "Mid-range colors (blue-green) represent returns around zero.",
        wrap=True,
        horizontalalignment="center",
        fontsize=12,
    )

    plt.tight_layout()
    plt.savefig(f"../data/generated/returns-heatmap.png")
    plt.close()


def create_rank_chart(data_frames, base_coin, start_date, end_date):
    # Calculate average returns for each coin
    avg_returns = {
        name: df["Returns"].mean() * 100 for name, df in data_frames.items()
    }  # Convert to percentage

    # Sort coins by average return
    sorted_returns = sorted(avg_returns.items(), key=lambda x: x[1], reverse=True)

    # Calculate final amount based on initial capital of $1000
    initial_capital = 50000
    final_amounts = {
        name: initial_capital * (1 + (return_pct / 100))
        for name, return_pct in avg_returns.items()
    }

    # Create a DataFrame for the sorted returns and final amounts
    sorted_returns_df = pd.DataFrame(sorted_returns, columns=["Coin", "Average Return"])
    sorted_returns_df["Final Amount ($)"] = sorted_returns_df["Coin"].map(final_amounts)

    # Create the bar chart
    plt.figure(figsize=(14, 18))  # Increase the figure size to make space for labels
    ax = sns.barplot(
        x="Average Return",
        y="Coin",
        data=sorted_returns_df,
        palette="viridis",
        edgecolor=".2",
    )

    plt.xlabel("Average Return (%)")
    plt.ylabel("Coin")
    plt.title(
        f"Ranking of Cryptocurrency Coins by Average Daily Return ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})\n"
        f"This bar chart shows how much $50k would grow from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}\n"
        f"by investing in each coin based on the average daily return."
    )
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    # Annotate bars with the final amount of money
    for index, row in sorted_returns_df.iterrows():
        ax.text(
            row["Average Return"] + 1.2,  # Adjust the offset as needed
            index,
            f"${row['Final Amount ($)']:.2f}",
            color="black",
            ha="center",
            va="center",
        )

    # Add explanation text
    plt.figtext(
        0.5,
        -0.05,
        "The average return is calculated as the mean of the daily returns for each coin over the analyzed period. "
        "A positive average return indicates the coin's price has generally increased, while a negative average return "
        "indicates a general decrease.",
        wrap=True,
        horizontalalignment="center",
        fontsize=10,
    )

    # Adjust y-axis label spacing for readability
    plt.yticks(fontsize=10, verticalalignment="center")

    plt.tight_layout()

    # Save the chart as an image file
    plt.savefig(f"../data/generated/returns-ranking-bar.png")
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Align cryptocurrency data based on a base coin's start date."
    )
    parser.add_argument(
        "-c",
        "--base-coin",
        required=True,
        help="Base coin to align all other coins' data with",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    filename = "../data/src/coingecko-coin-ids-daily.json"
    coin_names = load_coin_names(filename)
    data_frames = {}

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(process_coin_data, coin): coin
            for idx, coin in enumerate(coin_names)
        }

        # Fetch the base coin data to determine the alignment date
        base_coin_future = executor.submit(process_coin_data, args.base_coin)
        base_coin_data = base_coin_future.result()
        if base_coin_data is None:
            raise ValueError(f"No data found for base coin: {args.base_coin}")

        base_coin_start_date = base_coin_data.index.min()
        base_coin_end_date = base_coin_data.index.max()

        for future in as_completed(futures):
            coin_name = futures[future]
            result = future.result()
            if result is not None:
                data_frames[coin_name] = result

    aligned_data = align_datasets(data_frames, base_coin_start_date)
    volatility_data = calculate_volatility(aligned_data)

    with open(
        "../data/generated/" + args.base_coin + "-volatility-data.json", "w"
    ) as f:
        json.dump(volatility_data, f, indent=4)

    create_heatmap(aligned_data, args.base_coin)
    create_rank_chart(
        aligned_data, args.base_coin, base_coin_start_date, base_coin_end_date
    )


if __name__ == "__main__":
    main()

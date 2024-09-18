import argparse
import json
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans


def read_json_file(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def extract_prices(data, coin_id):
    prices = [price[1] for price in data["data"][coin_id]["prices"]]
    timestamps = [price[0] for price in data["data"][coin_id]["prices"]]
    return prices, timestamps


def calculate_profits(prices):
    results = []
    for buy_day in range(len(prices)):
        for sell_day in range(buy_day + 1, len(prices)):
            buy_price = prices[buy_day]
            sell_price = prices[sell_day]
            profit = sell_price - buy_price
            hold_time = sell_day - buy_day
            if profit > 0:
                results.append((buy_day, sell_day, profit, hold_time))
    return results


def convert_dates(df, timestamps):
    df["Buy Date"] = df["Buy Day"].apply(
        lambda x: datetime.utcfromtimestamp(timestamps[x] / 1000).strftime("%Y-%m-%d")
    )
    df["Sell Date"] = df["Sell Day"].apply(
        lambda x: datetime.utcfromtimestamp(timestamps[x] / 1000).strftime("%Y-%m-%d")
    )
    return df


def plot_heatmap(df, coin_id, top_trades_num):
    plt.figure(figsize=(12, 8))
    buy_heatmap_data = df.pivot_table(
        index=pd.to_datetime(df["Buy Date"]).dt.strftime("%Y-%m-%d"),
        columns=pd.to_datetime(df["Sell Date"]).dt.strftime("%Y-%m-%d"),
        values="Profit",
        fill_value=0,
    )

    # Create a custom color map
    cmap = sns.color_palette("rocket", as_cmap=True)

    start_date = str(
        df["Buy Date"].min().date()
    )  # Convert to string and only take the date part
    end_date = str(
        df["Sell Date"].max().date()
    )  # Convert to string and only take the date part

    sns.heatmap(
        buy_heatmap_data, cmap=cmap, vmin=0.0001
    )  # Set vmin to slightly above 0 to hide 0 values
    plt.title(
        f"{coin_id.upper()} | Heatmap of the top {top_trades_num} Profitable Trades | {start_date} to {end_date}"
    )
    plt.xlabel("Sell Date")
    plt.ylabel("Buy Date")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def plot_trends(trend_analysis, title, xlabel):
    plt.figure(figsize=(12, 6))
    plt.plot(
        trend_analysis["Year-Month"].astype(str),
        trend_analysis["Profit"],
        marker="o",
    )
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Average Profit")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()


def extract_day_of_month(df):
    df["Buy Day of Month"] = pd.to_datetime(df["Buy Date"]).dt.day
    return df


def apply_kmeans_clustering(df, n_clusters=4):
    kmeans = KMeans(n_clusters=n_clusters)
    df["Cluster"] = kmeans.fit_predict(df[["Buy Day of Month"]])
    return df


def plot_cluster_heatmap(df):
    plt.figure(figsize=(12, 8))
    cluster_heatmap_data = df.pivot_table(
        index=df["Cluster"],
        columns=df["Buy Day of Month"],
        values="Profit",
        aggfunc="mean",
        fill_value=0,
    )
    sns.heatmap(cluster_heatmap_data, cmap="YlGnBu", annot=True, fmt=".2f")
    plt.title("Heatmap of Average Profits by Buy Day of Month and Cluster")
    plt.xlabel("Buy Day of Month")
    plt.ylabel("Cluster")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze price data for a given coin.")
    parser.add_argument("-c", "--coin", required=True, help="Coin ID (e.g., gme)")
    parser.add_argument(
        "-f", "--file", required=True, help="Path to the price input JSON file"
    )

    args = parser.parse_args()

    file_path = args.file
    coin_id = args.coin

    data = read_json_file(file_path)

    prices, timestamps = extract_prices(data, coin_id)

    results = calculate_profits(prices)

    df = pd.DataFrame(results, columns=["Buy Day", "Sell Day", "Profit", "Hold Time"])
    df_sorted = df.sort_values(by="Profit", ascending=False)  # Sort by profit

    average_hold_time = df_sorted["Hold Time"].mean()
    print(f"Average Hold Time for Profitable Trades: {average_hold_time} days")

    df_sorted = convert_dates(df_sorted, timestamps)

    top_trades_num = 5000
    top_trades = df_sorted.head(top_trades_num)
    pd.set_option("display.max_rows", top_trades_num)
    print(f"Top {top_trades_num} Profitable Trades:")
    print(top_trades[["Buy Date", "Sell Date", "Profit", "Hold Time"]])

    top_trades.to_csv("top_profitable_trades.csv", index=False)

    profit_stats = top_trades["Profit"].describe()
    hold_time_stats = top_trades["Hold Time"].describe()

    print("\nProfit Statistics:")
    print(profit_stats)
    print("\nHold Time Statistics:")
    print(hold_time_stats)

    group_by_buy_date = (
        top_trades.groupby("Buy Date")
        .agg({"Profit": ["mean", "sum", "count"], "Hold Time": "mean"})
        .reset_index()
    )

    group_by_sell_date = (
        top_trades.groupby("Sell Date")
        .agg({"Profit": ["mean", "sum", "count"], "Hold Time": "mean"})
        .reset_index()
    )

    print("\nGroup by Buy Date:")
    print(group_by_buy_date.head())
    print("\nGroup by Sell Date:")
    print(group_by_sell_date.head())

    correlation = top_trades[["Profit", "Hold Time"]].corr()
    print("\nCorrelation between Profit and Hold Time:")
    print(correlation)

    top_trades["Buy Date"] = pd.to_datetime(top_trades["Buy Date"])
    top_trades["Sell Date"] = pd.to_datetime(top_trades["Sell Date"])

    time_series_buy = (
        top_trades.set_index("Buy Date")
        .resample("M")
        .agg({"Profit": "sum", "Hold Time": "mean"})
    )

    time_series_sell = (
        top_trades.set_index("Sell Date")
        .resample("M")
        .agg({"Profit": "sum", "Hold Time": "mean"})
    )

    print("\nTime Series Analysis on Buy Dates:")
    print(time_series_buy)
    print("\nTime Series Analysis on Sell Dates:")
    print(time_series_sell)

    top_trades["Buy Year-Month"] = top_trades["Buy Date"].dt.to_period("M")
    top_trades["Sell Year-Month"] = top_trades["Sell Date"].dt.to_period("M")

    trend_analysis_buy = (
        top_trades.groupby("Buy Year-Month")
        .agg({"Profit": "mean", "Hold Time": "mean"})
        .reset_index()
    )

    trend_analysis_sell = (
        top_trades.groupby("Sell Year-Month")
        .agg({"Profit": "mean", "Hold Time": "mean"})
        .reset_index()
    )

    print("\nTrend Analysis on Buy Year-Month:")
    print(trend_analysis_buy)

    print("\nTrend Analysis on Sell Year-Month:")
    print(trend_analysis_sell)

    plot_heatmap(top_trades, coin_id, top_trades_num)

    trend_analysis_buy.rename(columns={"Buy Year-Month": "Year-Month"}, inplace=True)
    # plot_trends(
    #     trend_analysis_buy, "Average Profit by Buy Year-Month", "Buy Year-Month"
    # )
    #
    # trend_analysis_sell.rename(columns={"Sell Year-Month": "Year-Month"}, inplace=True)
    # plot_trends(
    #     trend_analysis_sell, "Average Profit by Sell Year-Month", "Sell Year-Month"
    # )


if __name__ == "__main__":
    main()

"""
This script analyzes average hourly price swings for specific days of the week and month
based on historical price data for a given coin or stock. It performs the following steps:

1. Loads and preprocesses historical price data from a JSON file.
2. Resamples the data to hourly intervals and calculates hourly price swings.
3. Adds features such as day of the week, hour, and week of the month.
4. Filters the data to extract specific days of the week within a particular week of the month.
5. Calculates the trend, seasonality, and confidence intervals of the price swings.
6. Identifies and extracts the most recent actual price swings for the specified days.
7. Normalizes the data for consistent visualization.
8. Plots the average hourly price swings along with the trend, seasonality, and confidence intervals.
9. Optionally includes recent actual price swings for comparison in the plot.

The resulting plots highlight average hourly price swings for a specific day of the week
within a particular week of the month, along with recent actual price swings for comparison.

Usage: python3 price-swings-hourly.py -f <file_path> -c <coin> -i <index> -d <day> [--trend]

Arguments:
    -f, --file <file_path>   Path to the price data JSON file
    -c, --coin <coin>        Name of the coin or stock (e.g., AAPL, GME)
    -i, --index <index>      Week index within the month (1 for 1st week, 2 for 2nd week, etc.)
    -d, --day <day>          Day of the week (Mon, Tue, Wed, Thu, Fri, Sat, Sun)

Example:
    python3 price-swings-hourly.py -f ../data/src/gme-1-hour-2019-07-11-to-2024-07-09.json -i 2 -d Tue -c gme
"""

import argparse
import json

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy.ndimage import minimum_filter1d, maximum_filter1d
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler


def day_of_week_to_num(day):
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return days.index(day)


def find_recent_nth_days(df, day_num, week_index, n=8):  # Updated to 8
    df_filtered = df[df.index.weekday == day_num]
    df_filtered["week_of_month"] = df_filtered.index.to_series().apply(
        lambda x: (x.day - 1) // 7 + 1
    )
    df_filtered = df_filtered[df_filtered["week_of_month"] == week_index]
    unique_days = df_filtered.index.normalize().unique()
    recent_days = unique_days[-n:]
    print("Recent Days:", recent_days)  # Print unique recent days
    return recent_days if not recent_days.empty else None


def normalize_series(series, range_min=-1, range_max=1):
    scaler = MinMaxScaler(feature_range=(range_min, range_max))
    return scaler.fit_transform(series.values.reshape(-1, 1)).flatten()


def load_data(file_path, coin):
    with open(file_path, "r") as f:
        data = json.load(f)

    if coin not in data["data"]:
        raise ValueError(f"Coin or stock '{coin}' not found in data.")

    prices = data["data"][coin]["prices"]
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert("US/Eastern")
    df.set_index("timestamp", inplace=True)

    if coin.upper() in ["AAPL", "GOOG", "MSFT", "GME"]:
        df = df[df.index.dayofweek < 5]  # Exclude Saturdays and Sundays

    return df


def resample_and_calculate_swings(df):
    df_hourly = df.resample("H").last()
    df_hourly["swing"] = df_hourly["price"].pct_change()
    return df_hourly


def add_features(df_hourly):
    df_hourly["day_of_week"] = df_hourly.index.to_series().apply(lambda x: x.weekday())
    df_hourly["hour"] = df_hourly.index.hour
    df_hourly["week_of_month"] = df_hourly.index.to_series().apply(
        lambda x: (x.day - 1) // 7 + 1
    )
    return df_hourly


def filter_specific_days(df_hourly, day_num, week_index):
    return df_hourly[
        (df_hourly["day_of_week"] == day_num)
        & (df_hourly["week_of_month"] == week_index)
    ]


def calculate_trend(x, y, degree=2):
    poly = PolynomialFeatures(degree=degree)
    x_poly = poly.fit_transform(x)
    poly_reg = LinearRegression()
    poly_reg.fit(x_poly, y)
    trend = poly_reg.predict(x_poly)
    return trend


def calculate_confidence_interval(y, trend, multiplier=1.96):
    std_dev = np.std(y - trend)
    confidence_interval_upper = trend + multiplier * std_dev
    confidence_interval_lower = trend - multiplier * std_dev
    return confidence_interval_upper, confidence_interval_lower


def calculate_seasonality(series, window=3):
    return series.rolling(window=window, min_periods=1).mean()


def calculate_envelopes(series, size=3):
    lower_envelope = minimum_filter1d(series.values.flatten(), size=size)
    upper_envelope = maximum_filter1d(series.values.flatten(), size=size)
    return lower_envelope, upper_envelope


def plot_results(
    args,
    specific_days_pivot,
    trend,
    seasonality,
    lower_envelope,
    upper_envelope,
    recent_days_prices=None,
):
    normalized_avg_swing = normalize_series(specific_days_pivot)
    normalized_trend = normalize_series(pd.Series(trend.flatten()))
    normalized_seasonality = normalize_series(seasonality)
    normalized_lower_envelope = normalize_series(pd.Series(lower_envelope))
    normalized_upper_envelope = normalize_series(pd.Series(upper_envelope))

    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(5, 4, figure=fig)  # Change from 4x4 to 5x4 to add extra row
    main_ax = fig.add_subplot(gs[2:5, 1:3])  # Shift main_ax down by one row

    # Calculate swing intensities and normalize
    swing_intensities = np.abs(normalized_avg_swing)
    norm = mcolors.Normalize(vmin=swing_intensities.min(), vmax=swing_intensities.max())
    cmap = plt.get_cmap("viridis")

    # Plot each segment with a color based on intensity and add annotations
    for i in range(len(normalized_avg_swing) - 1):
        main_ax.plot(
            specific_days_pivot.index[i : i + 2],
            normalized_avg_swing[i : i + 2],
            marker="o",
            color=cmap(norm(swing_intensities[i])),
            linewidth=3,
        )
        main_ax.annotate(
            f"{specific_days_pivot.values[i][0]*100:.2f}%",  # Access scalar value
            (specific_days_pivot.index[i], normalized_avg_swing[i]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )

    main_ax.plot(
        specific_days_pivot.index,
        normalized_trend,
        label="Trend",
        linestyle="--",
        linewidth=1,
        alpha=0.6,
    )

    main_ax.plot(
        specific_days_pivot.index,
        normalized_seasonality,
        label="Seasonality",
        linestyle="-.",
        linewidth=1,
        alpha=0.6,
    )

    # main_ax.plot(
    #     specific_days_pivot.index,
    #     normalized_lower_envelope,
    #     label="Lower Bound",
    #     linestyle=":",
    #     color="red",
    #     linewidth=1,
    #     alpha=0.6,
    # )
    # main_ax.plot(
    #     specific_days_pivot.index,
    #     normalized_upper_envelope,
    #     label="Upper Bound",
    #     linestyle=":",
    #     color="blue",
    #     linewidth=1,
    #     alpha=0.6,
    # )

    # Plot the most recent nth day's data in the main plot
    # if recent_days_prices is not None and len(recent_days_prices) > 0:
    #     most_recent_day = recent_days_prices[-1]
    #     normalized_recent_day = normalize_series(most_recent_day)
    #     main_ax.plot(
    #         most_recent_day.index.hour,
    #         normalized_recent_day,
    #         label=f"Most Recent Actual ({most_recent_day.index.date[0]})",
    #         linestyle=":",
    #         linewidth=2,
    #         marker="x",
    #     )

    main_ax.set_title(
        f"\n"
        + f"{args.coin.upper()} | Average Hourly Price Swings | {args.day} of Week {args.index}\n"
        + f"Colors and percentages represent intensity of price swings. Great for directional bias and timing trades. "
        f"Not great for magnitude."
    )
    main_ax.set_xlabel("Hour of Day (EST)")
    main_ax.grid(True)
    main_ax.legend()

    am_pm_labels = []
    for hour in range(24):
        if hour == 0:
            label = "12 AM"
        elif hour < 12:
            label = f"{hour} AM"
        elif hour == 12:
            label = "12 PM"
        else:
            label = f"{hour - 12} PM"
        am_pm_labels.append(label)
    main_ax.set_xticks(ticks=np.arange(24))
    main_ax.set_xticklabels(am_pm_labels, rotation=90)
    main_ax.axes.get_yaxis().set_ticks([])  # Remove y ticks

    if recent_days_prices is not None:
        positions = [
            (0, 0),
            (0, 3),
            (1, 0),
            (1, 3),
            (0, 1),
            (0, 2),
            (1, 1),
            (1, 2),
        ]  # Updated for 8
        for i, recent_day_prices in enumerate(recent_days_prices):
            sub_ax = fig.add_subplot(gs[positions[i]])
            normalized_recent_day_prices = normalize_series(recent_day_prices)
            sub_ax.plot(
                recent_day_prices.index.hour,
                normalized_recent_day_prices,
                label=f"Most Recent Actual {i+1}",
                marker="x",
                linewidth=1,
            )
            sub_ax.set_title(f"Reference ({recent_day_prices.index.date[0]})")
            sub_ax.set_xticks(ticks=np.arange(24))
            sub_ax.set_xticklabels(am_pm_labels, rotation=90)
            sub_ax.grid(True)

    plt.tight_layout()
    plt.savefig(f"../data/generated/{args.coin}_{args.index}_{args.day}.png")
    plt.show()


def main(args):
    day_num = day_of_week_to_num(args.day)

    df = load_data(args.file, args.coin)
    df_hourly = resample_and_calculate_swings(df)
    df_hourly = add_features(df_hourly)

    specific_days = filter_specific_days(df_hourly, day_num, args.index)
    specific_days_pivot = specific_days.pivot_table(
        values="swing", index="hour", aggfunc=np.mean
    )

    x = specific_days_pivot.index.values.reshape(-1, 1)
    y = specific_days_pivot.values

    trend = calculate_trend(x, y)
    seasonality = calculate_seasonality(specific_days_pivot)
    lower_envelope, upper_envelope = calculate_envelopes(specific_days_pivot)

    recent_dates = find_recent_nth_days(df_hourly, day_num, args.index)
    recent_days_prices = []
    if recent_dates is not None:
        for recent_date in recent_dates:
            recent_day_data = df_hourly[df_hourly.index.date == recent_date.date()]
            recent_days_prices.append(recent_day_data["price"])

    plot_results(
        args,
        specific_days_pivot,
        trend,
        seasonality,
        lower_envelope,
        upper_envelope,
        recent_days_prices,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze average hourly price swings for specific days of the week and month"
    )
    parser.add_argument(
        "-f", "--file", type=str, required=True, help="Path to the price data JSON file"
    )
    parser.add_argument(
        "-c",
        "--coin",
        type=str,
        required=True,
        help="Name of the coin or stock (e.g., dogwifcoin, AAPL)",
    )
    parser.add_argument(
        "-i",
        "--index",
        type=int,
        required=True,
        help="Week index within the month (1 for 1st week, 2 for 2nd week, etc.)",
    )
    parser.add_argument(
        "-d",
        "--day",
        type=str,
        required=True,
        help="Day of the week (Mon, Tue, Wed, Thu, Fri, Sat, Sun)",
    )
    args = parser.parse_args()

    main(args)

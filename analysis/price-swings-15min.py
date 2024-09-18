import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler
from scipy.ndimage import minimum_filter1d, maximum_filter1d

#  python3 price-swings-15min.py -f ../data/src/gme-1-hour-2019-07-11-to-2024-07-09.json -i 2 -d Tue -c gme --trend


def day_of_week_to_num(day):
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return days.index(day)


def find_recent_nth_day(df, day_num, week_index):
    df_filtered = df[df.index.weekday == day_num]
    df_filtered["week_of_month"] = df_filtered.index.to_series().apply(
        lambda x: (x.day - 1) // 7 + 1
    )
    df_filtered = df_filtered[df_filtered["week_of_month"] == week_index]
    if not df_filtered.empty:
        return df_filtered.index.max()
    else:
        return None


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
    df_15min = df.resample("15T").last()
    df_15min["swing"] = df_15min["price"].pct_change()
    return df_15min


def add_features(df_15min):
    df_15min["day_of_week"] = df_15min.index.to_series().apply(lambda x: x.weekday())
    df_15min["time_of_day"] = df_15min.index.time
    df_15min["week_of_month"] = df_15min.index.to_series().apply(
        lambda x: (x.day - 1) // 7 + 1
    )
    return df_15min


def filter_specific_days(df_15min, day_num, week_index):
    return df_15min[
        (df_15min["day_of_week"] == day_num) & (df_15min["week_of_month"] == week_index)
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
    recent_day_prices=None,
):
    specific_days_pivot.index = specific_days_pivot.index.map(
        lambda x: x.hour + x.minute / 60.0
    )

    normalized_avg_swing = normalize_series(specific_days_pivot)
    normalized_trend = normalize_series(pd.Series(trend.flatten()))
    normalized_seasonality = normalize_series(seasonality)
    normalized_lower_envelope = normalize_series(pd.Series(lower_envelope))
    normalized_upper_envelope = normalize_series(pd.Series(upper_envelope))

    if recent_day_prices is not None:
        recent_day_prices.index = recent_day_prices.index.map(
            lambda x: x.hour + x.minute / 60.0
        )
        normalized_recent_day_prices = normalize_series(recent_day_prices)

    plt.figure(figsize=(15, 8))
    plt.plot(
        specific_days_pivot.index,
        normalized_avg_swing,
        marker="o",
        label="Forecast",
        linewidth=1.5,
    )

    if args.trend:
        plt.plot(
            specific_days_pivot.index,
            normalized_trend,
            label="Trend",
            linestyle="--",
            linewidth=1,
            alpha=0.6,
        )

    if args.seasonality:
        plt.plot(
            specific_days_pivot.index,
            normalized_seasonality,
            label="Seasonality",
            linestyle="-.",
            linewidth=1,
            alpha=0.6,
        )

    plt.plot(
        specific_days_pivot.index,
        normalized_lower_envelope,
        label="Lower Bound",
        linestyle=":",
        color="red",
        linewidth=1,
        alpha=0.6,
    )
    plt.plot(
        specific_days_pivot.index,
        normalized_upper_envelope,
        label="Upper Bound",
        linestyle=":",
        color="blue",
        linewidth=1,
        alpha=0.6,
    )

    if recent_day_prices is not None:
        plt.plot(
            recent_day_prices.index,
            normalized_recent_day_prices,
            label="Most Recent Actual",
            marker="x",
            color="green",
            linewidth=2,
        )

    plt.title(f"15-minute forecast for {args.day} for {args.coin.upper()}")
    plt.xlabel("Time of Day (EST)")
    plt.ylabel("Normalized Value")
    plt.grid(True)
    plt.legend()

    plt.xticks(rotation=90)

    plt.savefig(f"../data/generated/{args.coin}_{args.index}_{args.day}.png")
    plt.show()


def main(args):
    day_num = day_of_week_to_num(args.day)

    df = load_data(args.file, args.coin)
    df_15min = resample_and_calculate_swings(df)
    df_15min = add_features(df_15min)

    specific_days = filter_specific_days(df_15min, day_num, args.index)
    specific_days_pivot = specific_days.pivot_table(
        values="swing", index="time_of_day", aggfunc=np.mean
    )

    x = np.arange(len(specific_days_pivot)).reshape(-1, 1)
    y = specific_days_pivot.values

    trend = calculate_trend(x, y)
    confidence_interval_upper, confidence_interval_lower = (
        calculate_confidence_interval(y, trend)
    )
    seasonality = calculate_seasonality(specific_days_pivot)
    lower_envelope, upper_envelope = calculate_envelopes(specific_days_pivot)

    recent_date = find_recent_nth_day(df_15min, day_num, args.index)
    recent_day_prices = None
    if recent_date:
        recent_day_data = df_15min[df_15min.index.date == recent_date.date()]
        recent_day_prices = recent_day_data["price"]

    plot_results(
        args,
        specific_days_pivot,
        trend,
        seasonality,
        lower_envelope,
        upper_envelope,
        recent_day_prices,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze average 15-minute price swings for specific days of the week and month"
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
    parser.add_argument(
        "--trend", action="store_true", help="Include trend line in the plot"
    )
    parser.add_argument(
        "--seasonality", action="store_true", help="Include seasonality in the plot"
    )
    args = parser.parse_args()

    main(args)

"""
This script forecasts stock prices using Prophet after applying STL decomposition. It also saves the forecasts to a
JSON file and plots the results.

Usage:
    python3 stl-prophet.py -f <file> -c <asset_id> -t <time_frame> [-s <seasonal_period>]

Arguments:
    -f, --file              Path to the JSON file
    -c, --asset_id          Asset ID in the JSON file
    -t, --time_frame        Time frame for the forecast ('daily' or 'hourly')
    -s, --seasonal_period   Seasonal period for STL decomposition (default: 252 for daily, 24 for hourly)

Example:
    python3 stl-prophet.py -f ../data/src/gme-1-hour-2019-07-03-to-2024-05-01.json -c gme -t daily -s 252

Note:
    This STL + Prophet implementation EXCLUDES weekends when focusing on daily stock prices.
"""

import argparse
import json
from itertools import product

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.seasonal import STL


def load_data(file_path, asset_id):
    with open(file_path, "r") as f:
        data = json.load(f)
    prices = data["data"][asset_id]["prices"]
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.rename(columns={"timestamp": "ds", "price": "y"})
    return df.sort_values(by="ds")


def apply_stl_decomposition(df, seasonal_period):
    stl = STL(df["y"], period=seasonal_period)
    result = stl.fit()
    df["trend"] = result.trend
    df["seasonal"] = result.seasonal
    df["resid"] = result.resid
    return df


def define_param_grid(df):
    caps = [df["y"].max(), df["y"].max() * 1.25, df["y"].max() * 2]
    floors = [0]
    changepoint_prior_scales = [0.001, 0.01, 0.1, 0.5]
    seasonality_modes = ["additive", "multiplicative"]
    seasonality_prior_scales = [0.01, 0.1, 1.0, 5.0, 10.0]

    param_grid = {
        "cap": caps,
        "floor": floors,
        "changepoint_prior_scale": changepoint_prior_scales,
        "seasonality_mode": seasonality_modes,
        "seasonality_prior_scale": seasonality_prior_scales,
    }
    return [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]


def forecast_with_params(df, params, periods, freq):
    df["cap"] = params["cap"]
    df["floor"] = params["floor"]

    model = Prophet(
        changepoint_prior_scale=params["changepoint_prior_scale"],
        seasonality_mode=params["seasonality_mode"],
        seasonality_prior_scale=params["seasonality_prior_scale"],
    )
    model.fit(df[["ds", "y", "cap", "floor"]])

    future = model.make_future_dataframe(periods=periods, freq=freq)
    future["cap"] = params["cap"]
    future["floor"] = params["floor"]
    forecast = model.predict(future)
    future_forecast = forecast[forecast["ds"] > df["ds"].max()]
    return params, future_forecast


def save_forecasts_to_json(forecasts, output_file_path):
    output = {
        str(params): forecast.assign(
            ds=forecast["ds"].dt.strftime("%Y-%m-%d %H:%M:%S")
        )[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_dict(orient="records")
        for params, forecast in forecasts
    }
    with open(output_file_path, "w") as f:
        json.dump(output, f, indent=4)


def plot_forecasts(forecasts, asset_id, time_frame):
    plt.figure(figsize=(16, 9))  # Increase the figure size
    for params, forecast in forecasts:
        plt.plot(forecast["ds"], forecast["yhat"], label=str(params))

    # Format x-axis dates
    if time_frame == "daily":
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    elif time_frame == "hourly":
        plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=1))
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))

    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(
        f"{asset_id.upper()} | Price Forecast ({time_frame}) with STL Decomposition and Prophet"
    )
    plt.gcf().autofmt_xdate()  # Rotate and format date labels
    plt.savefig(f"../data/generated/{asset_id}-stl-prophet-forecast-{time_frame}.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Forecast stock prices using Prophet.")
    parser.add_argument("-f", "--file", required=True, help="Path to the JSON file")
    parser.add_argument(
        "-c", "--asset_id", required=True, help="Asset ID in the JSON file"
    )
    parser.add_argument(
        "-t",
        "--time_frame",
        default="daily",
        choices=["daily", "hourly"],
        help="Time frame for the forecast ('daily' or 'hourly') (default: daily)",
    )
    parser.add_argument(
        "-s",
        "--seasonal_period",
        type=int,
        default=7,
        help="Seasonal period for STL decomposition (default: 7)",
    )

    args = parser.parse_args()

    # Set defaults based on time frame if not provided
    seasonal_period = (
        args.seasonal_period
        if args.seasonal_period
        else (252 if args.time_frame == "daily" else 24)
    )
    periods = 7 if args.time_frame == "daily" else 24 * 7
    freq = "B" if args.time_frame == "daily" else "H"

    print(
        f"Seasonal period: {seasonal_period}. Forecasting {periods} periods ahead with frequency {freq}."
    )

    df = load_data(args.file, args.asset_id)
    df = apply_stl_decomposition(df, seasonal_period)

    param_grid = define_param_grid(df)

    forecasts = [
        forecast_with_params(df, params, periods, freq) for params in param_grid
    ]

    output_file_path = (
        f"../data/generated/{args.asset_id}-stl-prophet-forecast-{args.time_frame}.json"
    )
    save_forecasts_to_json(forecasts, output_file_path)

    plot_forecasts(forecasts, args.asset_id, args.time_frame)


if __name__ == "__main__":
    main()

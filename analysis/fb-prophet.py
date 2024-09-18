"""
Prophet implementation for predicting future prices of a cryptocurrency.

Usage:
    python3 fb-prophet.py -f <file> -c <coin> -i <interval> -s <steps>

Arguments:
    -f, --file          Path to the JSON file
    -c, --coin          Coin ID to extract data from
    -i, --interval      Data interval (minutely, hourly, daily)
    -s, --steps         Number of steps to forecast

Example:
    python3 fb-prophet.py -f ../data/src/gme-1-hour-2019-07-03-to-2024-05-01.json -c gme -i hourly -s 400

"""

import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet

# python3 fb-prophet.py -f ../data/src/gme-1-hour-2019-07-03-to-2024-05-01.json -c gme -i hourly -s 400


def load_data(file_path, coin_id):
    with open(file_path, "r") as file:
        data = json.load(file)
    timestamps = np.array([entry[0] for entry in data["data"][coin_id]["prices"]])
    prices = np.array([entry[1] for entry in data["data"][coin_id]["prices"]])
    dates = pd.to_datetime(timestamps, unit="ms")
    return pd.DataFrame({"ds": dates, "y": prices})


def fit_model(df, interval):
    model = Prophet(
        changepoint_prior_scale=0.5,
        changepoint_range=0.8,
        mcmc_samples=1000,
    )
    model.fit(df)
    return model


def make_predictions(model, steps, interval):
    future_dates = model.make_future_dataframe(
        periods=steps,
        freq=("H" if interval == "hourly" else "T" if interval == "minutely" else "D"),
    )
    forecast = model.predict(future_dates)

    # Ensure no negative predictions
    forecast["yhat"] = forecast["yhat"].apply(lambda x: max(x, 0))
    forecast["yhat_lower"] = forecast["yhat_lower"].apply(lambda x: max(x, 0))
    forecast["yhat_upper"] = forecast["yhat_upper"].apply(lambda x: max(x, 0))

    return forecast


def plot_results(df, forecast, interval):
    plt.figure(figsize=(12, 6))

    # First chart: Observed prices and predictions
    plt.subplot(2, 1, 1)
    plt.plot(df["ds"], df["y"], "o", label="Observed prices")
    plt.plot(forecast["ds"], forecast["yhat"], label="Predicted prices")
    plt.fill_between(
        forecast["ds"],
        forecast["yhat_lower"],
        forecast["yhat_upper"],
        alpha=0.2,
        label="Uncertainty interval",
    )
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.title("Observed Prices and Predictions for Historical Data")

    # Second chart: Predictions only
    plt.subplot(2, 1, 2)
    forecast_only = forecast[forecast["ds"] > df["ds"].max()]
    plt.plot(forecast_only["ds"], forecast_only["yhat"], label="Predicted prices")
    plt.fill_between(
        forecast_only["ds"],
        forecast_only["yhat_lower"],
        forecast_only["yhat_upper"],
        alpha=0.2,
        label="Uncertainty interval",
    )
    plt.xlabel(f"Time ({interval})")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.title("Future Price Predictions")

    plt.tight_layout()
    plt.show()


def main(file_path, coin_id, interval, steps):
    df = load_data(file_path, coin_id)
    model = fit_model(df, interval)
    forecast = make_predictions(model, steps, interval)
    plot_results(df, forecast, interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict future prices using Prophet.")
    parser.add_argument(
        "-f", "--file", type=str, required=True, help="Path to the JSON file"
    )
    parser.add_argument(
        "-c", "--coin", type=str, required=True, help="Coin ID to extract data from"
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=str,
        default="daily",
        choices=["minutely", "hourly", "daily"],
        help="Data interval (minutely, hourly, daily)",
    )
    parser.add_argument(
        "-s", "--steps", type=int, required=True, help="Number of steps to forecast"
    )
    args = parser.parse_args()

    main(args.file, args.coin, args.interval, args.steps)

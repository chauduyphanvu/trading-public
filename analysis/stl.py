"""
This script performs STL decomposition followed by Prophet forecasting on stock/cryptocurrency price data.

Usage:
    python3 stl.py -c <base_coin> [--start-date <start_date>] [--file <file>] [--steps <steps>]

Arguments:
    -c, --base-coin    ID of the coin (e.g., 'bitcoin', 'ethereum')
    --start-date       Start date in 'YYYY-MM-DD' format (optional)
    -f, --file          Path to the input JSON file
    -steps             Number of forecast steps

Example:
    python3 stl.py -c gme --f ../data/src/bitcoin-1-hour-2019-06-17-to-2024-06-13.json -steps 365
"""

import argparse
import concurrent
import json
import matplotlib.pyplot as plt
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.seasonal import STL


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Perform STL decomposition and Prophet forecasting on cryptocurrency price data."
    )
    parser.add_argument(
        "-c",
        "--base-coin",
        required=True,
        help="ID of the coin (e.g., 'bitcoin', 'ethereum')",
    )
    parser.add_argument(
        "--start-date", type=str, required=False, help="Start date in YYYY-MM-DD format"
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=False,
        default="../data/src/gme-1-hour-2019-06-17-to-2024-06-13.json",
        help="Path to the input JSON file",
    )
    parser.add_argument(
        "-steps",
        type=int,
        required=False,
        default=365,
        help="Number of forecast steps",
    )
    return parser.parse_args()


def load_data(file_path, coin_id, start_date=None):
    with open(file_path, "r") as file:
        raw_data = json.load(file)
    prices = raw_data["data"][coin_id]["prices"]
    data = pd.DataFrame(prices, columns=["timestamp", "price"])
    data["timestamp"] = pd.to_datetime(data["timestamp"], unit="ms")
    if start_date:
        data = data[data["timestamp"] >= start_date]
    data.set_index("timestamp", inplace=True)
    return data


def preprocess_data(data):
    full_range = pd.date_range(start=data.index.min(), end=data.index.max(), freq="D")
    return data.reindex(full_range).interpolate(method="linear")


def perform_stl_decomposition(ts):
    stl = STL(ts, seasonal=7, robust=True)
    return stl.fit()


def save_results(result, forecasts, configs, coin_id, timestamps):
    data_to_save = {
        "timestamps": timestamps,
        "trend": result.trend.tolist(),
        "seasonal": result.seasonal.tolist(),
        "residual": result.resid.tolist(),
        "forecasts": {
            "_".join([f"{key}={config[key]}" for key in config]): forecast[
                "yhat"
            ].tolist()
            for config, forecast in zip(configs, forecasts)
        },
    }
    filename = f"../data/generated/{coin_id}-stl-prophet-forecast.json"
    with open(filename, "w") as file:
        json.dump(data_to_save, file, indent=4)


def forecast_prophet(
    ts,
    steps,
    cap,
    floor,
    changepoint_prior_scale,
    seasonality_mode,
    seasonality_prior_scale,
):
    df = ts.reset_index().rename(columns={"index": "ds", "price": "y"})
    df["cap"] = cap
    df["floor"] = floor  # Set the floor value

    model = Prophet(
        growth="logistic",
        seasonality_mode=seasonality_mode,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
    )
    model.fit(df)
    future = model.make_future_dataframe(periods=steps)
    future["cap"] = cap  # Ensure future dataframe also has 'cap' column
    future["floor"] = floor  # Ensure future dataframe also has 'floor' column

    forecast = model.predict(future)
    forecast["yhat"] = forecast["yhat"].clip(
        lower=floor
    )  # Optionally ensure no negative values

    return forecast[["ds", "yhat"]].set_index("ds")[-steps:]


def plot_forecasts(base_coin_id, ts, result, forecasts, steps):
    plt.figure(figsize=(12, 6))
    # plt.plot(ts, label="Historical Prices", color="blue")
    # plt.plot(result.trend, label="Trend", color="orange")
    # plt.plot(result.seasonal, label="Seasonal", color="green")
    # plt.plot(result.resid, label="Residual", color="red")
    colors = ["purple", "magenta", "cyan", "brown", "gray", "yellow", "black", "pink"]
    for i, forecast in enumerate(forecasts):
        plt.plot(
            forecast.index,
            forecast["yhat"],
            label=f"Forecast Config {i + 1}",
            color=colors[i % len(colors)],
        )
    plt.legend()
    plt.title(
        f"{base_coin_id.upper()} STL Decomposition and Prophet Forecasts for {steps} steps"
    )
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.grid()
    plt.show()


def run_forecast(config, ts, steps, base_coin):
    print(f"Running configuration for {base_coin} with config: {config}")
    return forecast_prophet(
        ts,
        steps,
        config["cap"],
        config["floor"],
        config["changepoint_prior_scale"],
        config["seasonality_mode"],
        config["seasonality_prior_scale"],
    )


def main():
    args = parse_arguments()
    start_date = args.start_date
    data = load_data(args.file, args.base_coin, start_date)
    processed_data = preprocess_data(data)
    ts = processed_data["price"]

    timestamps = processed_data.index.astype("int64") // 1e6  # Convert 'ns' to 'ms'
    result = perform_stl_decomposition(ts)

    forecast_steps = args.steps

    # TODO: Create coin-specific configurations (specifically for those >1Y old and those <1Y old)
    # TODO: For recent meme coins, we will likely need to tune some hyperparameters that Prophet's docs recommend not
    #  tuning
    caps = [ts.max(), ts.max() * 1.25, ts.max() * 2]
    floors = [0]  # Setting floor to 0 to avoid negative forecasts
    changepoint_prior_scales = [0.001, 0.01, 0.1, 0.5]
    seasonality_modes = ["additive", "multiplicative"]
    seasonality_prior_scales = [0.01, 0.1, 1.0, 5.0, 10.0]

    configs = [
        {
            "cap": cap,
            "floor": floor,
            "changepoint_prior_scale": changepoint_prior_scale,
            "seasonality_mode": seasonality_mode,
            "seasonality_prior_scale": seasonality_prior_scale,
        }
        for cap in caps
        for floor in floors
        for changepoint_prior_scale in changepoint_prior_scales
        for seasonality_mode in seasonality_modes
        for seasonality_prior_scale in seasonality_prior_scales
    ]

    total_combinations = len(configs)
    print(f"Total number of combinations: {total_combinations}")

    forecasts = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(run_forecast, config, ts, forecast_steps, args.base_coin)
            for config in configs
        ]
        for future in concurrent.futures.as_completed(futures):
            forecasts.append(future.result())

    save_results(result, forecasts, configs, args.base_coin, timestamps.tolist())
    plot_forecasts(args.base_coin, ts, result, forecasts, forecast_steps)


if __name__ == "__main__":
    main()

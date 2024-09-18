import argparse
import json
from datetime import timedelta

import numpy as np
import pywt

import boto3
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
from tqdm import tqdm

# Feature Requests:
#
# SMA 50 might not be useful. Reconsider removing it.
# Seasonal Decomposition: Decompose the time series into seasonal, trend, and residual components to capture underlying patterns.
# VWAP (Volume Weighted Average Price), ATR (Average True Range)
# Lagged Features: Add more lagged features (e.g., lag_20, lag_30) to capture longer-term dependencies.
# Wavelet Transform: Apply wavelet transforms to decompose the time series data into different frequency components, which can then be used as features.
# Fourier Transform: Use Fourier transforms to capture cyclic patterns in the data.


def load_data(file_path, coin):
    with open(file_path, "r") as file:
        data = json.load(file)
    prices = pd.DataFrame(data["data"][coin]["prices"], columns=["timestamp", "price"])
    volumes = pd.DataFrame(
        data["data"][coin]["volumes"], columns=["timestamp", "volume"]
    )
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], unit="ms")
    volumes["timestamp"] = pd.to_datetime(volumes["timestamp"], unit="ms")
    prices.set_index("timestamp", inplace=True)
    volumes.set_index("timestamp", inplace=True)
    data = prices.join(volumes)
    return data


def add_wavelet_features(data, wavelet="db1", level=3):
    coeffs = pywt.wavedec(data["price"], wavelet, level=level)
    for i, coeff in enumerate(coeffs):
        # TODO: This pads the coefficients with zeros to match the length of the original data.
        # Might be something to consider changing in the future.
        coeff_padded = np.pad(
            coeff, (0, len(data) - len(coeff)), "constant", constant_values=(0,)
        )
        data[f"wavelet_coeff_{i}"] = coeff_padded[: len(data)]
    return data


def create_features(data, interval):
    if interval == "5min":
        window5, window10, window14, window20, window26, window50, window3 = (
            60,
            120,
            168,
            240,
            312,
            600,
            36,
        )  # Approximate windows for 5-min
    elif interval == "15min":
        window5, window10, window14, window20, window26, window50, window3 = (
            20,
            40,
            56,
            80,
            104,
            200,
            12,
        )  # Approximate windows for 15-min
    elif interval == "hourly":
        window5, window10, window14, window20, window26, window50, window3 = (
            5,
            10,
            14,
            20,
            26,
            50,
            3,
        )
    elif interval == "daily":
        window5, window10, window14, window20, window26, window50, window3 = (
            5,
            10,
            14,
            20,
            26,
            50,
            3,
        )
    else:
        raise ValueError("Invalid interval. Use 'daily', 'hourly', '15min', or '5min'.")

    data["return"] = data["price"].pct_change()
    data["sma_5"] = data["price"].rolling(window=window5).mean()
    data["sma_10"] = data["price"].rolling(window=window10).mean()
    # data["sma_50"] = data["price"].rolling(window=window50).mean()  # Calculate MA50
    data["day_of_week"] = data.index.dayofweek
    data["month"] = data.index.month
    data["hour_of_day"] = data.index.hour  # New feature
    data["minute_of_hour"] = data.index.minute  # New feature
    for lag in range(1, 11):  # Adding more lag features
        data[f"lag_{lag}"] = data["price"].shift(lag)

    # Volatility
    data["volatility_5"] = data["return"].rolling(window=window5).std()

    # EMA
    data["ema_5"] = data["price"].ewm(span=window5).mean()
    data["ema_10"] = data["price"].ewm(span=window10).mean()

    # RSI
    delta = data["price"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window14).mean()
    rs = gain / loss
    data["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = data["price"].ewm(span=12).mean()
    ema_26 = data["price"].ewm(span=26).mean()
    data["macd"] = ema_12 - ema_26
    data["signal_line"] = data["macd"].ewm(span=9).mean()

    # Bollinger Bands
    data["bollinger_up"] = (
        data["price"].rolling(window=window20).mean()
        + data["price"].rolling(window=window20).std() * 2
    )
    data["bollinger_down"] = (
        data["price"].rolling(window=window20).mean()
        - data["price"].rolling(window=window20).std() * 2
    )

    # ATR
    data["tr"] = data["price"].diff().abs()
    data["atr"] = data["tr"].rolling(window=window14).mean()

    # Momentum
    data["momentum"] = data["price"].diff(window10)

    # Stochastic Oscillator
    data["stoch_k"] = (
        100
        * (data["price"] - data["price"].rolling(window=window14).min())
        / (
            data["price"].rolling(window=window14).max()
            - data["price"].rolling(window=window14).min()
        )
    )
    data["stoch_d"] = data["stoch_k"].rolling(window=window3).mean()

    # Volume features
    data["volume_change"] = data["volume"].pct_change()
    data["volume_sma_5"] = data["volume"].rolling(window=window5).mean()
    data["volume_sma_10"] = data["volume"].rolling(window=window10).mean()

    # Add wavelet features
    data = add_wavelet_features(data)

    data.dropna(inplace=True)
    return data


def train_test_split(data, train_size_ratio=0.8):
    train_size = int(len(data) * train_size_ratio)
    train, test = data.iloc[:train_size], data.iloc[train_size:]
    return train, test


def train_model(X_train, y_train):
    param_space = {
        "n_estimators": Integer(100, 200),
        "max_features": Categorical(["sqrt", "log2"]),
        "max_depth": Categorical([10, 20, None]),
        "min_samples_split": Integer(2, 10),
        "min_samples_leaf": Integer(1, 4),
        "bootstrap": Categorical([True, False]),
    }
    rf = RandomForestRegressor(random_state=42)
    tscv = TimeSeriesSplit(n_splits=5)

    n_iter = 50  # Number of iterations for BayesSearchCV
    pbar = tqdm(
        total=n_iter,
        desc="Hyperparameter tuning (Bayesian Optimization)",
        position=0,
        leave=True,
    )

    def on_step(result):
        pbar.update(1)

    bayes_search = BayesSearchCV(
        estimator=rf,
        search_spaces=param_space,
        cv=tscv,
        n_jobs=-1,  # Use all available CPU cores for concurrency
        verbose=0,
        n_iter=n_iter,
        optimizer_kwargs={"base_estimator": "GP", "n_initial_points": 10},
    )

    bayes_search.fit(X_train, y_train, callback=on_step)

    pbar.close()

    return bayes_search.best_estimator_


def make_predictions(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred


def forecast_future(model, data, features, steps, interval):
    max_window_size = 10  # This should be the maximum window size used in features

    if interval == "daily":
        delta = timedelta(days=1)
    elif interval == "hourly":
        delta = timedelta(hours=1)
    elif interval == "15min":
        delta = timedelta(minutes=15)
    elif interval == "5min":
        delta = timedelta(minutes=5)
    else:
        raise ValueError("Invalid interval. Use 'daily', 'hourly', '15min', or '5min'.")

    future_timestamps = [data.index[-1] + delta * i for i in range(1, steps + 1)]
    future_data = pd.DataFrame(index=future_timestamps, columns=data.columns)

    required_history = max_window_size + steps
    last_known_data = data.iloc[-required_history:].copy()

    future_data.iloc[0] = last_known_data.iloc[-1]

    predicted_prices = []

    for i in range(steps):
        if i < len(last_known_data):
            future_data.iloc[i] = last_known_data.iloc[i]
        else:
            combined_data = pd.concat([data, future_data.iloc[:i]])

            # Update features for future data points
            future_data["return"].iloc[i] = combined_data["price"].pct_change().iloc[-1]
            future_data["sma_5"].iloc[i] = (
                combined_data["price"].rolling(window=5).mean().iloc[-1]
            )
            future_data["sma_10"].iloc[i] = (
                combined_data["price"].rolling(window=10).mean().iloc[-1]
            )
            for lag in range(1, 11):
                future_data[f"lag_{lag}"].iloc[i] = (
                    combined_data["price"].shift(lag).iloc[-1]
                )
            future_data["day_of_week"].iloc[i] = future_timestamps[i].dayofweek
            future_data["month"].iloc[i] = future_timestamps[i].month

            future_data["volatility_5"].iloc[i] = (
                combined_data["return"].rolling(window=5).std().iloc[-1]
            )
            future_data["ema_5"].iloc[i] = (
                combined_data["price"].ewm(span=5).mean().iloc[-1]
            )
            future_data["ema_10"].iloc[i] = (
                combined_data["price"].ewm(span=10).mean().iloc[-1]
            )
            delta = combined_data["price"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            future_data["rsi"].iloc[i] = 100 - (100 / (1 + rs.iloc[-1]))

            ema_12 = combined_data["price"].ewm(span=12).mean()
            ema_26 = combined_data["price"].ewm(span=26).mean()
            future_data["macd"].iloc[i] = ema_12.iloc[-1] - ema_26.iloc[-1]
            future_data["signal_line"].iloc[i] = (
                future_data["macd"].ewm(span=9).mean().iloc[-1]
            )

            future_data["bollinger_up"].iloc[i] = (
                combined_data["price"].rolling(window=20).mean().iloc[-1]
                + combined_data["price"].rolling(window=20).std().iloc[-1] * 2
            )
            future_data["bollinger_down"].iloc[i] = (
                combined_data["price"].rolling(window=20).mean().iloc[-1]
                - combined_data["price"].rolling(window=20).std().iloc[-1] * 2
            )

            future_data["tr"].iloc[i] = combined_data["price"].diff().abs().iloc[-1]
            future_data["atr"].iloc[i] = (
                combined_data["tr"].rolling(window=14).mean().iloc[-1]
            )

            future_data["momentum"].iloc[i] = combined_data["price"].diff(10).iloc[-1]

            future_data["stoch_k"].iloc[i] = (
                100
                * (
                    combined_data["price"].iloc[-1]
                    - combined_data["price"].rolling(window=14).min().iloc[-1]
                )
                / (
                    combined_data["price"].rolling(window=14).max().iloc[-1]
                    - combined_data["price"].rolling(window=14).min().iloc[-1]
                )
            )
            future_data["stoch_d"].iloc[i] = (
                future_data["stoch_k"].rolling(window=3).mean().iloc[-1]
            )

            future_data["volume_change"].iloc[i] = (
                combined_data["volume"].pct_change().iloc[-1]
            )
            future_data["volume_sma_5"].iloc[i] = (
                combined_data["volume"].rolling(window=5).mean().iloc[-1]
            )
            future_data["volume_sma_10"].iloc[i] = (
                combined_data["volume"].rolling(window=10).mean().iloc[-1]
            )

            # Impute missing values before prediction
            future_features = future_data.iloc[i][features]

            if not future_features.isnull().any():
                future_price = model.predict([future_features])[0]
                future_data["price"].iloc[i] = future_price
                predicted_prices.append(future_price)
                print(f"Step {i + 1}/{steps}: Predicted price = {future_price}")
            else:
                missing_features = future_features[
                    future_features.isnull()
                ].index.tolist()
                print(
                    f"Step {i + 1}/{steps}: Missing features {missing_features}, unable to predict"
                )
                predicted_prices.append(None)

        print(f"Step {i + 1}: Future data row: {future_data.iloc[i]}")

    return future_data


def plot_results(full_data, future_data, steps, coin, interval, config_file):
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d (%H:%M:%S)")
    interval_display = (
        "days"
        if interval.lower() == "daily"
        else (
            "hours"
            if interval.lower() == "hourly"
            else (
                "15-minute intervals"
                if interval.lower() == "15min"
                else "5-minute intervals"
            )
        )
    )
    plt.figure(figsize=(14, 14))

    # Plot predictions against historical data
    plt.subplot(2, 1, 1)
    plt.plot(
        full_data.index[len(full_data) - len(test) :],
        full_data["price"].iloc[len(full_data) - len(test) :],
        label="Actual Price",
        color="blue",
    )
    plt.plot(
        full_data.index[len(full_data) - len(test) :],
        full_data["Predicted"].iloc[len(full_data) - len(test) :],
        label="Predicted Price",
        color="red",
        linestyle="--",
    )
    plt.legend()
    plt.title(f"{coin.upper()} | Predictions against Actual Prices (Testing)")
    plt.xlabel("Time")
    plt.ylabel("Price (USD)")
    plt.gca().set_yticklabels([])  # Remove y-axis tick labels

    # Plot future predictions
    plt.subplot(2, 1, 2)
    plt.plot(
        future_data.index, future_data["price"], label="Forecasted Price", color="green"
    )
    plt.legend()
    plt.title(
        f"{coin.upper()} | Predictions for next {steps} {interval_display} | Generated on {timestamp}"
    )
    plt.xlabel(f"Time ({interval.capitalize()})")
    plt.ylabel("Price (USD)")
    plt.xticks(rotation=45)
    plt.gca().set_yticklabels([])  # Remove y-axis tick labels

    # Set the x-axis limits to ensure all future data is shown
    plt.xlim([future_data.index[0], future_data.index[-1]])

    # Adjust the layout with some space between the subplots
    plt.tight_layout(h_pad=4.0)

    # Save the plot as a PNG file
    timestamp_for_filename = pd.Timestamp.now().strftime("%Y%m%d%H%M%S")
    file_path = f"../data/generated/{coin.lower()}-random-forest-next-{steps}-{interval_display}-{timestamp_for_filename}.png"
    plt.savefig(file_path)

    # Upload the plot to S3
    s3_key = f"Trading/random_forest/{coin.lower()}-random-forest-next-{steps}-{interval_display}-{timestamp_for_filename}.png"
    upload_to_s3(file_path, s3_key, config_file)

    plt.show()


def upload_to_s3(file_path, s3_key, config_file):
    with open(config_file, "r") as f:
        credentials = json.load(f)["s3"]

    s3 = boto3.client(
        "s3",
        aws_access_key_id=credentials["aws_access_key_id"],
        aws_secret_access_key=credentials["aws_secret_access_key"],
        region_name=credentials["region"],
    )

    try:
        bucket_name = credentials["bucket_name"]
        s3.upload_file(file_path, bucket_name, s3_key)
        print(f"File {file_path} uploaded to S3 bucket {bucket_name}. Key: {s3_key}")
    except Exception as e:
        print(f"Error uploading file to S3: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict future prices using RandomForestRegressor"
    )
    parser.add_argument(
        "-f", "--file", type=str, required=True, help="Path to the price data JSON file"
    )
    parser.add_argument("-c", "--coin", type=str, required=True, help="Coin ID name")
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        required=True,
        help="Number of steps to predict",
    )
    parser.add_argument(
        "-i",
        "--interval",
        type=str,
        required=True,
        choices=["daily", "hourly", "15min", "5min"],
        help="Prediction interval: 'daily', 'hourly', '15min', or '5min'",
    )
    args = parser.parse_args()
    config_file = "../data/src/config.json"

    prices = load_data(args.file, args.coin)
    prices = create_features(prices, args.interval)

    features = (
        [
            "return",
            "sma_5",
            "sma_10",
            "day_of_week",
            "month",
            "hour_of_day",
            "minute_of_hour",
        ]
        + [f"lag_{lag}" for lag in range(1, 11)]
        + [
            "volatility_5",
            "ema_5",
            "ema_10",
            "rsi",
            "macd",
            "signal_line",
            "bollinger_up",
            "bollinger_down",
            "atr",
            "momentum",
            "stoch_k",
            "stoch_d",
            "volume_change",
            "volume_sma_5",
            "volume_sma_10",
        ]
        + [f"wavelet_coeff_{i}" for i in range(4)]  # Add wavelet coefficients
    )
    target = "price"

    train, test = train_test_split(prices)
    X_train, y_train = train[features], train[target]
    X_test, y_test = test[features], test[target]

    model = train_model(X_train, y_train)
    y_pred = make_predictions(model, X_test)

    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    full_data = pd.concat([train, test])
    full_data["Predicted"] = None
    full_data["Predicted"].iloc[len(train) :] = y_pred

    future_data = forecast_future(model, prices, features, args.steps, args.interval)
    print(f"Forecast starts from {future_data.head()}")
    print(f"Forecast ends at {future_data.tail()}")
    print(f"Forecasted prices: {future_data['price'].values}")

    plot_results(
        full_data, future_data, args.steps, args.coin, args.interval, config_file
    )

import json
import pandas as pd
import numpy as np
from arch import arch_model
import matplotlib.pyplot as plt
import argparse
from scipy.optimize import minimize


def main(json_path, horizon, coin_id):
    # Load the price data from the JSON file
    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract the price data for the specified coin
    prices = data["data"][coin_id]["prices"]
    df = pd.DataFrame(prices, columns=["timestamp", "price"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    # Get the latest price
    current_price = df["price"].iloc[-1]

    # Calculate the log returns and rescale
    df["log_return"] = np.log(df["price"] / df["price"].shift(1)) * 100
    df = df.dropna()

    # Fit an ARMA-GARCH model to the returns
    model = arch_model(
        df["log_return"], mean="AR", lags=1, vol="Garch", p=1, q=1, rescale=True
    )

    # Define a custom callback function to set optimizer options
    def custom_callback(params, res, *args, **kwargs):
        return minimize(
            res, params, method="L-BFGS-B", options={"maxiter": 1000, "tol": 1e-6}
        )

    model_fit = model.fit(disp="off", options={"maxiter": 1000, "ftol": 1e-6})

    # Check model summary
    print(model_fit.summary())

    # Forecast future returns
    forecasts = model_fit.forecast(horizon=horizon)

    # Extract the forecasted mean and variance
    forecasted_mean = forecasts.mean.iloc[-1, :]
    forecasted_variance = forecasts.variance.iloc[-1, :]

    # Check forecasted mean and variance values
    print("Forecasted Mean:\n", forecasted_mean)
    print("Forecasted Variance:\n", forecasted_variance)

    # Calculate the confidence intervals for the forecasted returns
    confidence_interval = 1.96 * np.sqrt(forecasted_variance)
    forecasted_returns_lower = forecasted_mean - confidence_interval
    forecasted_returns_upper = forecasted_mean + confidence_interval

    # Calculate future prices
    future_prices = [current_price]
    for r_t in forecasted_mean:
        next_price = future_prices[-1] * (1 + r_t / 100)
        future_prices.append(next_price)

    future_prices = pd.Series(future_prices[1:])  # Remove initial current price

    # Print the future prices
    print(f"Future prices for the next {horizon} steps: \n{future_prices} \n")

    # Convert forecasted log returns to price predictions
    last_price = df["price"].iloc[-1]
    forecasted_prices = last_price * np.exp(np.cumsum(forecasted_mean / 100))
    forecasted_prices_lower = last_price * np.exp(
        np.cumsum(forecasted_returns_lower / 100)
    )
    forecasted_prices_upper = last_price * np.exp(
        np.cumsum(forecasted_returns_upper / 100)
    )

    # Prepare the forecast results
    forecast_dates = pd.date_range(
        start=df.index[-1] + pd.Timedelta(hours=1), periods=horizon, freq="H"
    )
    forecast_results = pd.DataFrame(
        {
            "Date": forecast_dates,
            "Forecasted_Returns": forecasted_mean,
            "Lower_CI": forecasted_returns_lower,
            "Upper_CI": forecasted_returns_upper,
            "Forecasted_Prices": forecasted_prices,
            "Lower_CI_Price": forecasted_prices_lower,
            "Upper_CI_Price": forecasted_prices_upper,
        }
    )

    # Plot the results
    plt.figure(figsize=(12, 6))

    # Plot historical prices
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df["price"], label="Historical Prices")
    plt.plot(
        forecast_results["Date"],
        forecast_results["Forecasted_Prices"],
        label="Forecasted Prices",
        linestyle="--",
    )
    plt.fill_between(
        forecast_results["Date"],
        forecast_results["Lower_CI_Price"],
        forecast_results["Upper_CI_Price"],
        color="k",
        alpha=0.1,
        label="Confidence Interval",
    )
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"{coin_id.upper()} Price Forecast using ARMA-GARCH")
    plt.legend()
    plt.grid(True)

    # Plot forecasted returns
    plt.subplot(2, 1, 2)
    plt.plot(
        forecast_results["Date"],
        forecast_results["Forecasted_Returns"],
        label="Forecasted Returns",
    )
    plt.fill_between(
        forecast_results["Date"],
        forecast_results["Lower_CI"],
        forecast_results["Upper_CI"],
        color="k",
        alpha=0.1,
        label="Confidence Interval",
    )
    plt.xlabel("Date")
    plt.ylabel("Log Returns")
    plt.title(f"Forecasted Returns using ARMA-GARCH for {coin_id.upper()}")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    # plt.show()

    # Save the forecast results to a CSV file
    forecast_results.to_csv("forecast_results.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARMA-GARCH Forecasting Script")
    parser.add_argument(
        "-f",
        "--json_path",
        type=str,
        required=True,
        help="Path to the JSON file containing the price data",
    )
    parser.add_argument(
        "-s", "--horizon", type=int, required=True, help="Number of periods to forecast"
    )
    parser.add_argument(
        "-c", "--coin_id", type=str, required=True, help="Coin ID to forecast"
    )
    args = parser.parse_args()

    main(args.json_path, args.horizon, args.coin_id)

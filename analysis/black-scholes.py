import json

import numpy as np
from scipy.stats import norm


# Load stock price data from JSON file
def load_stock_data(json_file):
    with open(json_file, "r") as file:
        data = json.load(file)
    prices = [entry[1] for entry in data["data"]["gme"]["prices"]]
    return prices


# Calculate historical volatility (standard deviation of log returns)
def calculate_historical_volatility(prices):
    log_returns = np.diff(np.log(prices))
    volatility = np.std(log_returns) * np.sqrt(len(log_returns))  # Annualize volatility
    return volatility


# Black-Scholes model formula
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == "call":
        option_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put option
        option_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return option_price


# Example usage
json_file = "../data/src/gme-15-minute-2019-08-23-to-2024-08-21.json"
prices = load_stock_data(json_file)
volatility = calculate_historical_volatility(prices)

# Parameters for Black-Scholes model
S = prices[-1]  # Current stock price (latest in your data)
K = 22.55  # Strike price (example)
T = 30 / 365  # Time to maturity in years (example: 30 days)
r = 0.05  # Risk-free interest rate (example: 5%)
sigma = volatility  # Historical volatility

# Calculate option price
call_price = black_scholes(S, K, T, r, sigma, option_type="call")
put_price = black_scholes(S, K, T, r, sigma, option_type="put")

print(f"Call Option Price: {call_price}")
print(f"Put Option Price: {put_price}")

import argparse
import json
import time

import requests

API_KEY = ""
BASE_URL = "https://api.polygon.io/v3/trades"


def fetch_trades(ticker, limit):
    print(f"Fetching trades for {ticker} with a limit of {limit}...")

    url = f"{BASE_URL}/{ticker}?limit={limit}&apiKey={API_KEY}"
    response = requests.get(url)
    trades = response.json()["results"]

    formatted_data = {
        "fetched_at": int(time.time()),
        "endpoint": "trades",
        "ticker": ticker,
        "limit": limit,
        "data": trades,
    }

    trades_json = json.dumps(formatted_data, indent=4)

    filename = f"../data/src/{ticker}-trades-{limit}-{int(time.time())}.json"
    with open(filename, "w") as f:
        f.write(trades_json)
    print(f"Data saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch trades for a cryptocurrency from Polygon.io"
    )
    parser.add_argument(
        "-c",
        "--ticker",
        type=str,
        required=True,
        help="The ticker symbol of the cryptocurrency (e.g., X:BTCUSD)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="The number of trades to fetch (default: 10)",
    )

    args = parser.parse_args()

    fetch_trades(args.ticker.upper(), args.limit)

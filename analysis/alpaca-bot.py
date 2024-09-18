import argparse
import concurrent.futures
import json
import logging
import signal
from datetime import datetime, timedelta

import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - [Alpaca Bot] %(message)s"
)

# Keep track of open buy and sell order IDs to cancel them if needed
buy_order_ids = []
sell_order_ids = []

# Keep track of processed signals to avoid duplicate processing
processed_signals = set()

# Keep track of how much profit has been made so far to avoid overtrading
starting_capital = None
profit_target = None
current_profit = 0

# Alpaca API credentials
alpaca_base_url = ""
alpaca_api_key = ""
alpaca_api_secret = ""


def get_headers():
    return {
        "accept": "application/json",
        "content-type": "application/json",
        "APCA-API-KEY-ID": alpaca_api_key,
        "APCA-API-SECRET-KEY": alpaca_api_secret,
    }


def handle_response(response):
    try:
        response.raise_for_status()
        return response.json()
    except requests.HTTPError as http_err:
        logging.error(f"❌ HTTP error occurred: {http_err}")
    except Exception as err:
        logging.error(f"❌ Other error occurred: {err}")
    return None


def api_request(method, endpoint, payload=None):
    url = f"{alpaca_base_url}/{endpoint}"
    headers = get_headers()
    response = requests.request(method, url, json=payload, headers=headers)

    logging.info(f"{method.upper()} {endpoint} response: {response.text}")
    return handle_response(response)


def get_account():
    return api_request("GET", "account")


def get_open_positions():
    return api_request("GET", "positions")


def cancel_order(order_id):
    return api_request("DELETE", f"orders/{order_id}")


def create_buy_order(
    symbol, limit_price, order_type="limit", time_in_force="gtc", extended_hours=True
):
    """
    Creates a buy order for the given symbol at the specified limit price. Cancels ALL existing buy orders before
    placing a new one.

    Note: Crypto buys need "gtc" time in force. For stocks, flip it to "day".

    :param symbol: The symbol to buy
    :param limit_price: The limit price to buy at
    :param order_type: The order type (default: "limit")
    :param time_in_force: The time in force for the order (default: "gtc")
    :param extended_hours: Whether to allow extended hours trading (default: True)

    :return: The response from the API
    """

    # By canceling all existing buy orders, we're prioritizing the latest signal in a group of consecutive buy signals.
    # To prioritize the earliest signal, comment out the following block.
    # Cancel all existing buy orders concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(cancel_order, order_id) for order_id in buy_order_ids
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            logging.info(f"Cancelled buy order result: {result}")

    buy_order_ids.clear()

    account = get_account()
    logging.info(f"Account details: {account}")

    # buying_power = float(account.get("buying_power", 0))
    buying_power = float(account.get("cash", 0))
    logging.info(f"Buying power (cash): ${buying_power}")

    # This assumes we're buying 100% of our buying power
    qty = buying_power / limit_price
    logging.info(f"Qty to buy with $${buying_power}: {qty} shares/coins")

    if qty < 0.0001:
        logging.warning("⚠️ Insufficient funds to place buy order. Skipping.")
        return {"error": "Insufficient funds to place buy order"}

    order_payload = {
        "side": "buy",
        "type": order_type,
        "time_in_force": time_in_force,
        "symbol": symbol,
        "qty": qty,
        "limit_price": limit_price,
        "extended_hours": extended_hours,
    }

    logging.info(f"Buy order payload: {order_payload}")
    response = api_request("POST", "orders", order_payload)
    if response:
        buy_order_ids.append(response["id"])
    return response


def create_sell_order(
    symbol, limit_price, order_type="limit", time_in_force="day", extended_hours=True
):
    """
    Creates a sell order for the given symbol at the specified limit price. Cancels ALL existing sell orders before
    placing a new one.

    :param symbol: The symbol to sell
    :param limit_price: The limit price to sell at
    :param order_type: The order type (default: "limit")
    :param time_in_force: The time in force for the order (default: "day")
    :param extended_hours: Whether to allow extended hours trading (default: True)

    :return: The response from the API
    """

    # By canceling all existing sell orders, we're prioritizing the latest signal in a group of consecutive sell
    # signals. To prioritize the earliest signal, comment out the following block.
    # Cancel all existing sell orders concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(cancel_order, order_id) for order_id in sell_order_ids
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            logging.info(f"Cancelled sell order result: {result}")

    sell_order_ids.clear()

    open_positions = get_open_positions()

    if not open_positions:
        logging.warning(
            "⚠️ Skipping sell order. No open positions found. Nothing to sell."
        )
        return {"error": "No open positions found"}

    # This assumes we're selling all shares/coins of the given symbol
    qty_available = next(
        (
            int(pos["qty_available"])
            for pos in open_positions
            if pos["symbol"] == symbol
        ),
        0,
    )

    if qty_available == 0:
        logging.warning("⚠️ No shares held to place sell order")
        return {"error": "No shares held"}
    else:
        logging.info(f"Qty available to sell: {qty_available}")

    order_payload = {
        "side": "sell",
        "type": order_type,
        "time_in_force": time_in_force,
        "symbol": symbol,
        "qty": qty_available,
        "limit_price": limit_price,
        "extended_hours": extended_hours,
    }

    logging.info(f"Sell order payload: {order_payload}")
    response = api_request("POST", "orders", order_payload)
    if response:
        sell_order_ids.append(response["id"])
    return response


@app.route("/alpaca-bot", methods=["POST"])
def trade():
    global starting_capital, profit_target, current_profit

    data = request.json
    required_fields = ["type", "symbol", "price", "timestamp"]
    if not all(field in data for field in required_fields):
        logging.error("❌ Invalid signal data. Missing required fields.")
        return jsonify({"error": "Invalid signal data"}), 400

    signal_type, symbol, limit_price, signal_time = (
        data["type"],
        data["symbol"],
        data["price"],
        datetime.strptime(data["timestamp"], "%Y-%m-%d %H:%M:%S"),
    )

    # Check if the signal has already been processed
    signal_key = (signal_type, symbol, signal_time)
    if signal_key in processed_signals:
        logging.info(f"⚠️ Signal {signal_key} has already been processed. Skipping.")
        return jsonify({"error": "Signal already processed"}), 200

    if starting_capital is None:
        account = get_account()
        starting_capital = float(account.get("buying_power", 0))
        profit_target = starting_capital * 0.02
        logging.info(
            f"Starting capital set to ${starting_capital}. Profit target is ${profit_target}."
        )

    if current_profit >= profit_target:
        logging.info("💰 Daily profit target reached. Stopping trading for the day.")
        return jsonify({"error": "Daily profit target reached. Trading stopped."}), 200

    if datetime.utcnow() - signal_time > timedelta(seconds=120):
        logging.warning("⚠️ Received stale signal. Ignoring.")
        return jsonify({"error": "Stale signal"}), 400

    if signal_type == "buy":
        logging.info(f"Attempting to buy ${symbol} at ${limit_price}/unit...")
        order_response = create_buy_order(symbol, limit_price=limit_price)
    elif signal_type == "sell":
        logging.info(f"Attempting to sell ${symbol} at ${limit_price}/unit...")
        order_response = create_sell_order(symbol, limit_price=limit_price)

        # Update the current profit after a sell order
        if order_response and "filled_avg_price" in order_response:
            sell_price = float(order_response["filled_avg_price"])
            qty = int(order_response["qty"])
            buy_price = limit_price
            current_profit += (sell_price - buy_price) * qty
            logging.info(f"Updated current profit: ${current_profit}")

    # Mark the signal as processed
    processed_signals.add(signal_key)

    return jsonify(order_response)


@app.route("/account", methods=["GET"])
def account():
    return jsonify(get_account())


@app.route("/positions", methods=["GET"])
def positions():
    return jsonify(get_open_positions())


@app.route("/cancel_order", methods=["POST"])
def cancel_order_endpoint():
    data = request.json
    if "order_id" not in data:
        logging.error("❌ Invalid order data")
        return jsonify({"error": "Invalid order data"}), 400
    return jsonify(cancel_order(data["order_id"]))


def handle_termination_signal(signum, frame):
    logging.info(
        "Received bot termination signal. Performing cleanup before exiting..."
    )
    logging.info("Cancelling all open orders before exiting...")

    # Cancel all existing buy orders concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(cancel_order, order_id) for order_id in buy_order_ids
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            logging.info(f"Cancelled buy order result: {result}")

    # Cancel all existing sell orders concurrently
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(cancel_order, order_id) for order_id in sell_order_ids
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            logging.info(f"Cancelled sell order result: {result}")

    buy_order_ids.clear()
    sell_order_ids.clear()

    logging.info(f"Final profit for the session: ${current_profit}.")
    logging.info("Exiting bot with code 0...")
    exit(0)


signal.signal(signal.SIGTERM, handle_termination_signal)
signal.signal(signal.SIGINT, handle_termination_signal)


def main():
    global alpaca_base_url, alpaca_api_key, alpaca_api_secret

    parser = argparse.ArgumentParser(description="Alpaca Trading Bot")
    parser.add_argument(
        "--mode",
        choices=["paper", "live"],
        required=True,
        help="Choose the mode: 'paper' for paper trading, 'live' for live trading",
    )
    args = parser.parse_args()

    config_file_path = "../data/src/config.json"
    with open(config_file_path) as config_file:
        config = json.load(config_file)

    if args.mode == "paper":
        alpaca_api_key = config["alpaca"]["api_key_paper"]
        alpaca_api_secret = config["alpaca"]["secret_paper"]
        alpaca_base_url = "https://paper-api.alpaca.markets/v2"
        logging.info("⚠️ Running in paper trading mode...")
    else:
        alpaca_api_key = config["alpaca"]["api_key"]
        alpaca_api_secret = config["alpaca"]["secret"]
        alpaca_base_url = "https://api.alpaca.markets/v2"
        logging.info("⚠️ Running in live trading mode...")

    app.run(port=5000)


if __name__ == "__main__":
    main()

import argparse
import json
import logging
import uuid
from datetime import datetime

import pytz
from coinbase.rest import RESTClient
from flask import Flask, request

app = Flask(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [Coinbase Trading Bot] %(message)s",
)


def load_config():
    with open("../data/src/config.json") as config_file:
        config = json.load(config_file)
    return config


config = load_config()

coinbase_client = RESTClient(
    api_key=config["coinbase"]["api_key"],
    api_secret=config["coinbase"]["api_secret"],
    timeout=5,
)

asset_id = None

# Coinbase trading fees. Not accurate, but close enough. Adjust as needed.
BUY_FEE_RATE = 0.0006
SELL_FEE_RATE = 0.0008

# Default to limit orders for safety
buy_order_type = "limit"
sell_order_type = "limit"

# Signals that have been processed so we don't double-process them (TA is performed at x intervals and may call the
# bot with the same signal repeatedly even if the signal is old)
processed_buy_signals = []
processed_sell_signals = []

# The most recent action that was successfully executed so we can avoid consecutive actions
latest_action_succeeded = None

# Order IDs for open buy and sell orders
open_buy_order_ids = []
open_sell_order_ids = []


@app.route("/coinbase-bot", methods=["POST"])
def trade():
    data = request.json
    signal_type = data["type"]
    price = float(data["price"])
    timestamp = datetime.strptime(data["timestamp"], "%Y-%m-%d %H:%M:%S")

    # Make timestamp timezone-aware
    timestamp = pytz.timezone("US/Eastern").localize(timestamp)

    if signal_type == "buy":
        handle_buy_signal(price, timestamp, buy_order_type)
    elif signal_type == "sell":
        handle_sell_signal(price, timestamp, sell_order_type)

    return "Signal received", 200


def handle_buy_signal(price, timestamp, order_type):
    global asset_id, latest_action_succeeded, processed_buy_signals, signal_distances

    if timestamp in processed_buy_signals:
        logging.error(f"⚠️ Skipping buy order placement. Signal already processed.")

    product_id = f"{asset_id.upper()}-USD"
    accounts = coinbase_client.get_accounts(limit=100)
    accounts_json = json.loads(json.dumps(accounts))

    usd_bal = 0
    for account in accounts_json["accounts"]:
        if account["currency"] == "USD":
            usd_bal = float(account["available_balance"]["value"])
            logging.info(f"USD balance available for trading: ${usd_bal}")
            break

    limit_price = price * (1 - BUY_FEE_RATE)
    token_qty = usd_bal / limit_price
    token_qty = token_qty * 0.99
    base_size_str = "{:.8f}".format(token_qty)

    if token_qty < 0.0001:
        logging.error(f"⚠️ Skipping buy order placement. Insufficient USD balance.")
        return

    place_order(
        coinbase_client,
        product_id,
        "buy",
        order_type,
        base_size=base_size_str,
        price=str(limit_price) if order_type == "limit" else price,
    )
    processed_buy_signals.append(timestamp)


def handle_sell_signal(price, timestamp, order_type):
    global asset_id, latest_action_succeeded, processed_sell_signals, signal_distances

    if timestamp in processed_sell_signals:
        logging.error(f"⚠️ Skipping sell order placement. Signal already processed.")
        return

    product_id = f"{asset_id.upper()}-USD"
    accounts = coinbase_client.get_accounts(limit=100)
    accounts_json = json.loads(json.dumps(accounts))

    limit_price = price * (1 + SELL_FEE_RATE)
    token_bal = 0

    for account in accounts_json["accounts"]:
        if account["currency"] == asset_id.upper():
            token_bal = float(account["available_balance"]["value"])
            formatted_token_bal = "{:.8f}".format(token_bal)
            logging.info(
                f"${asset_id.upper()} balance available for selling: {formatted_token_bal}"
            )

    if token_bal > 0.0001:
        place_order(
            coinbase_client,
            product_id,
            "sell",
            order_type,
            base_size=formatted_token_bal,
            price=str(limit_price) if order_type == "limit" else price,
        )
        processed_sell_signals.append(timestamp)
    else:
        logging.error(
            f"⚠️ Skipping sell order placement. Insufficient ${asset_id.upper()} balance."
        )


def cancel_open_orders(client, order_type):
    global open_buy_order_ids, open_sell_order_ids
    try:
        if order_type == "buy" and open_buy_order_ids:
            client.cancel_orders(order_ids=open_buy_order_ids)
            logging.info(f"ℹ️ Canceled open buy order(s): {open_buy_order_ids}")
            open_buy_order_ids = []
        elif order_type == "sell" and open_sell_order_ids:
            client.cancel_orders(order_ids=open_sell_order_ids)
            logging.info(f"ℹ️ Canceled open sell order(s): {open_sell_order_ids}")
            open_sell_order_ids = []
    except Exception as e:
        logging.error(f"❌ Exception occurred while canceling open orders: {e}")


def place_order(client, product_id, side, order_type, base_size, price=None):
    global latest_action_succeeded, open_buy_order_ids, open_sell_order_ids, fee_adjusted_base_size
    order = None

    try:
        cancel_open_orders(client, side)

        client_order_id = str(uuid.uuid4())
        base_size = float(base_size)

        logging.info(
            f"Attempting to place Coinbase {side} order for {product_id} with base size {base_size}..."
        )

        if side == "buy":
            fee_adjusted_base_size = float(base_size * (1 - BUY_FEE_RATE))
        elif side == "sell":
            fee_adjusted_base_size = float(base_size * (1 - SELL_FEE_RATE))

        if side == "buy":
            if order_type == "limit":
                price = float(price)
                logging.info(f"Pinging Coinbase to place a limit buy order...")
                order = client.limit_order_ioc_buy(
                    client_order_id=client_order_id,
                    product_id=product_id,
                    base_size="{:.8f}".format(fee_adjusted_base_size),
                    limit_price="{:.2f}".format(price * (1 - BUY_FEE_RATE)),
                )
            elif order_type == "market":
                logging.info(f"Pinging Coinbase to place a market buy order...")
                order = client.market_order_buy(
                    client_order_id=client_order_id,
                    product_id=product_id,
                    quote_size="{:.2f}".format(base_size * price),
                )

        elif side == "sell":
            if order_type == "limit":
                price = float(price)
                logging.info(f"Pinging Coinbase to place a limit sell order...")
                order = client.limit_order_gtc_sell(
                    client_order_id=client_order_id,
                    product_id=product_id,
                    base_size="{:.8f}".format(fee_adjusted_base_size),
                    limit_price="{:.2f}".format(price * (1 + SELL_FEE_RATE)),
                )
            elif order_type == "market":
                logging.info(f"Pinging Coinbase to place a market sell order...")
                order = client.market_order_sell(
                    client_order_id=client_order_id,
                    product_id=product_id,
                    base_size="{:.8f}".format(base_size),
                )

        if order and order.get("success", False):
            logging.info(
                f"✅ Coinbase {side.capitalize()} order successfully placed: {order}."
            )
            latest_action_succeeded = "buy" if side == "buy" else "sell"
            if side == "buy":
                open_buy_order_ids.append(order["order_id"])
                if len(open_buy_order_ids) > 100:
                    open_buy_order_ids = open_buy_order_ids[-100:]
            elif side == "sell":
                open_sell_order_ids.append(order["order_id"])
                if len(open_sell_order_ids) > 100:
                    open_sell_order_ids = open_sell_order_ids[-100:]
        else:
            failure_reason, error_details = parse_order_placement_failure_reason(order)
            logging.error(
                f"❌ Coinbase {side.capitalize()} order failed to be placed. Reason: {failure_reason}. Details: {error_details}"
            )

    except Exception as e:
        logging.error(f"❌ Exception occurred while placing order: {e}")


def parse_order_placement_failure_reason(response):
    if not response.get("success", False):
        error_response = response.get("error_response", {})
        failure_reason = (
            error_response.get("message", "Unknown reason") or "None provided"
        )
        error_details = error_response.get("error_details", "") or "None provided"
        return failure_reason, error_details
    return None, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start the trading bot")
    parser.add_argument("-c", "--coin", type=str, required=True, help="Coin/stock ID")
    parser.add_argument(
        "-bt",
        "--buy_order_type",
        type=str,
        choices=["limit", "market"],
        default="limit",
        help="Buy order type: limit or market",
    )
    parser.add_argument(
        "-st",
        "--sell_order_type",
        type=str,
        choices=["limit", "market"],
        default="limit",
        help="Sell order type: limit or market",
    )
    args = parser.parse_args()
    asset_id = args.coin
    buy_order_type = args.buy_order_type
    sell_order_type = args.sell_order_type

    logging.info(f"Starting the Coinbase trading bot for {asset_id}...")
    logging.info(f"Buy order type: {buy_order_type}")
    logging.info(f"Sell order type: {sell_order_type}")

    # If asset ID starts with `x:` then it's a cryptocurrency. Drop the `x:` prefix and the `usd` suffix.
    if asset_id.startswith("x:"):
        asset_id = asset_id[2:]  # Drop the `x:` prefix
        if asset_id.endswith("usd"):
            asset_id = asset_id[:-3]  # Drop the `usd` suffix

    app.run(port=5000)

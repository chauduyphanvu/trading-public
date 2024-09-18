import argparse
import glob
import gzip
import json
import logging
import os
import platform
import subprocess
import threading
import time
from datetime import datetime, timedelta
from typing import List

import boto3
import pandas as pd
import pandas_ta as ta
import pytz
import requests
from polygon import WebSocketClient
from polygon.websocket.models import WebSocketMessage, Market
from tabulate import tabulate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [Technical Analysis] %(message)s",
)


# Define the trading hours (in EST)
TRADING_START_HOUR = 9  # 9 AM EST
TRADING_END_HOUR = 16  # 4 PM EST
EST = pytz.timezone("US/Eastern")


def load_config():
    with open("../data/src/config.json") as config_file:
        config = json.load(config_file)

    return config


config = load_config()
price_data = pd.DataFrame()
last_analysis_time = None
asset_id = None
analysis_lock = threading.Lock()


def is_within_trading_hours(timestamp):
    # Check if the timestamp is naive (lacking timezone information)
    if timestamp.tzinfo is None:
        # Localize the timestamp to UTC
        timestamp = timestamp.tz_localize(pytz.UTC)
    # Convert UTC timestamp to EST
    est_timestamp = timestamp.tz_convert(EST)
    return TRADING_START_HOUR <= est_timestamp.hour < TRADING_END_HOUR


def filter_signals(signals):
    signals = signals[signals.index.map(is_within_trading_hours)]
    return signals


def stream_aggs(msgs: List[WebSocketMessage], interval: str):
    """
    Stream price data and perform technical analysis every second or minute.

    :param msgs: List of WebSocket messages
    :param interval: Streaming interval ("second" or "minute")
    :return: None
    """
    global price_data, last_analysis_time
    new_data = pd.DataFrame(
        {
            "timestamp": [
                pd.to_datetime(m.start_timestamp, unit="ms", utc=True).tz_convert(
                    "US/Eastern"
                )
                for m in msgs
            ],
            "price": [m.close for m in msgs],
        }
    )
    new_data.set_index("timestamp", inplace=True)
    price_data = pd.concat([price_data, new_data])
    price_data = price_data[~price_data.index.duplicated(keep="first")]
    min_data_points = 20 if interval == "second" else 5

    if len(price_data) >= min_data_points:
        analysis_thread = threading.Thread(target=perform_analysis)
        analysis_thread.start()

    logging.info(f"Last streamed price: ${price_data['price'].iloc[-1]}")


def stream_price(interval: str):
    global asset_id
    is_crypto = asset_id.startswith("x:")
    asset_id = get_clean_symbol(asset_id)

    if interval == "minute":
        asset_symbol = (
            f"XA.{asset_id.upper()}-USD" if is_crypto else f"AM.{asset_id.upper()}"
        )
    else:  # "second"
        asset_symbol = (
            f"XAS.{asset_id.upper()}-USD" if is_crypto else f"A.{asset_id.upper()}"
        )

    market_type = Market.Crypto if is_crypto else Market.Stocks
    api_key = config["polygon"]["api_key"]

    websocket_client = WebSocketClient(api_key=api_key, market=market_type)
    websocket_client.subscribe(asset_symbol)
    websocket_client.run(lambda msgs: stream_aggs(msgs, interval))


def fetch_data(coin, start_date, multiplier, timespan):
    """
    Fetch historical data into a local JSON file (not streaming). This is a one-time operation per script run.

    :param coin:
    :param start_date:
    :param multiplier:
    :param timespan:
    :return:
    """
    command = "python3" if platform.system() != "Windows" else "python"
    command_path = os.path.abspath("polygon-fetch.py")
    subprocess.run(
        [
            command,
            command_path,
            "-c",
            coin,
            "--start-date",
            start_date,
            "--multiplier",
            multiplier,
            "--timespan",
            timespan,
        ]
    )


def calculate_indicators(data):
    # Simple Moving Average (SMA): Calculates the average of the closing prices over a specified period.
    data["SMA"] = ta.sma(data["price"], length=14)

    # Exponential Moving Average (EMA): Similar to SMA, but gives more weight to recent prices.
    data["EMA"] = ta.ema(data["price"], length=14)

    # Relative Strength Index (RSI): Measures the speed and change of price movements. RSI values above 70 indicate overbought conditions, and below 30 indicate oversold conditions.
    data["RSI"] = ta.rsi(data["price"], length=14)

    # Bollinger Bands (BB): Consists of a middle band (SMA) and two outer bands (standard deviations away from SMA). Prices near the upper band are considered overbought, while prices near the lower band are considered oversold.
    bbands = ta.bbands(data["price"], length=20)
    if bbands is not None:
        data["upper_band"] = bbands["BBU_20_2.0"]
        data["middle_band"] = bbands["BBM_20_2.0"]
        data["lower_band"] = bbands["BBL_20_2.0"]

    # Average True Range (ATR): Measures market volatility by decomposing the entire range of an asset price for that period.
    data["ATR"] = ta.atr(data["price"], data["price"], data["price"], length=14)

    # Stochastic Oscillator: Compares a particular closing price of a security to a range of its prices over a certain period. It is used to generate overbought and oversold trading signals.
    stoch = ta.stoch(data["price"], data["price"], data["price"])
    data["Stoch_K"] = stoch["STOCHk_14_3_3"]
    data["Stoch_D"] = stoch["STOCHd_14_3_3"]

    # On-Balance Volume (OBV): Uses volume flow to predict changes in stock price. It adds volume on up days and subtracts volume on down days.
    data["OBV"] = ta.obv(data["price"], data["price"])

    # Commodity Channel Index (CCI): Measures the current price level relative to an average price level over a given period. High values indicate an overbought condition, and low values indicate an oversold condition.
    data["CCI"] = ta.cci(data["price"], data["price"], data["price"], length=14)

    # Volume Weighted Average Price (VWAP): Gives the average price a security has traded at throughout the day, based on both volume and price.
    data["VWAP"] = ta.vwap(data["price"], data["price"], data["price"], data["volume"])

    # Additional indicators
    # Moving Average Convergence Divergence (MACD): Shows the relationship between two moving averages of a security’s price. The MACD is calculated by subtracting the 26-period EMA from the 12-period EMA.
    macd = ta.macd(data["price"], fast=12, slow=26, signal=9)
    data["MACD"] = macd["MACD_12_26_9"]
    data["MACD_Signal"] = macd["MACDs_12_26_9"]
    data["MACD_Hist"] = macd["MACDh_12_26_9"]

    # Average Directional Index (ADX): Measures the strength of a trend. Values above 20 indicate a strong trend.
    adx = ta.adx(data["price"], data["price"], data["price"], length=14)
    data["ADX"] = adx["ADX_14"]
    data["ADX+DI"] = adx["DMP_14"]
    data["ADX-DI"] = adx["DMN_14"]

    # Money Flow Index (MFI): Uses both price and volume to measure buying and selling pressure. MFI values above 80 are considered overbought, and below 20 are considered oversold.
    data["MFI"] = ta.mfi(
        data["price"], data["price"], data["price"], data["volume"], length=14
    )

    # TRIX: A momentum oscillator that shows the percent rate of change of a triple exponentially smoothed moving average.
    trix = ta.trix(data["price"], length=14)
    data["TRIX"] = trix["TRIX_14_9"]
    data["TRIX_signal"] = trix["TRIXs_14_9"]

    # True Strength Index (TSI): Measures the strength of a trend by smoothing price changes.
    tsi = ta.tsi(data["price"], long=25, short=13)
    data["TSI"] = tsi["TSI_13_25_13"]
    data["TSI_signal"] = tsi["TSIs_13_25_13"]

    # Williams %R: A momentum indicator that measures overbought and oversold levels.
    data["Williams %R"] = ta.willr(
        data["price"], data["price"], data["price"], length=14
    )

    # Ultimate Oscillator: Combines short, intermediate, and long-term price action into one value.
    data["Ultimate Oscillator"] = ta.uo(
        data["price"], data["price"], data["price"], s=7, m=14, len=28
    )

    data.fillna(method="ffill", inplace=True)
    data.fillna(method="bfill", inplace=True)
    return data


def generate_signals(data):
    # Ensure that the index is timezone-aware
    if data.index.tzinfo is None:
        data.index = data.index.tz_localize(pytz.UTC)

    signals = pd.DataFrame(index=data.index)
    signals["price"] = data["price"]
    signals["signal"] = 0.0

    indicators_for_signals = []

    # Combine signals from multiple indicator checks
    rsi_buy, rsi_sell = check_rsi(data)
    bb_buy, bb_sell = check_bollinger_bands(data)
    cci_buy, cci_sell = check_cci(data)
    vwap_buy, vwap_sell = check_vwap(data)
    ema_buy, ema_sell = check_ema(data)
    # adr_buy, adr_sell = check_adr(data)
    adx_buy, adx_sell = check_adx(data)
    mfi_buy, mfi_sell = check_mfi(data, tolerance=0.5)
    trix_buy, trix_sell = check_trix(data)
    tsi_buy, tsi_sell = check_tsi(data)
    williams_r_buy, williams_r_sell = check_williams_r(data)
    ultimate_oscillator_buy, ultimate_oscillator_sell = check_ultimate_oscillator(data)
    stoch_buy, stoch_sell = check_stochastic(data)

    # Note: Combinations are timeframe-specific. If timeframe changes (e.g., from 1M to 1D), re-evaluate the conditions.
    # MFI is usually the strictest, so OR it with other indicators (or groups of indicators)
    buy_conditions = mfi_buy  # | rsi_buy | (williams_r_buy & ultimate_oscillator_buy & stoch_buy & tsi_buy)
    sell_conditions = (
        mfi_sell | rsi_sell | (williams_r_sell & ultimate_oscillator_sell & stoch_sell)
    )

    signals.loc[buy_conditions, "signal"] = 1
    signals.loc[sell_conditions, "signal"] = -1
    signals["positions"] = signals["signal"].diff()

    buy_signals = signals[signals["signal"] == 1]
    sell_signals = signals[signals["signal"] == -1]

    # Uncomment to filter out signals outside trading hours
    buy_signals = filter_signals(buy_signals)
    sell_signals = filter_signals(sell_signals)

    indicators_for_signals.extend(
        ["MFI", "TSI", "Ultimate Oscillator", "Williams %R", "Stochastic", "RSI"]
    )

    return buy_signals, sell_signals, indicators_for_signals


def check_vwap(data):
    buy_conditions = data["price"] < data["VWAP"]
    sell_conditions = data["price"] > data["VWAP"]
    return buy_conditions, sell_conditions


def check_rsi(data):
    buy_conditions = data["RSI"] < 30
    sell_conditions = data["RSI"] > 70
    return buy_conditions, sell_conditions


def check_macd(data):
    buy_conditions = data["MACD"] > data["MACD_Signal"]
    sell_conditions = data["MACD"] < data["MACD_Signal"]
    return buy_conditions, sell_conditions


def check_bollinger_bands(data):
    buy_conditions = data["price"] < data["lower_band"]
    sell_conditions = data["price"] > data["upper_band"]
    return buy_conditions, sell_conditions


def check_cci(data):
    buy_conditions = data["CCI"] < -100
    sell_conditions = data["CCI"] > 100
    return buy_conditions, sell_conditions


def check_ema(data):
    buy_conditions = data["EMA"] > data["price"]
    sell_conditions = data["EMA"] < data["price"]
    return buy_conditions, sell_conditions


def check_adr(data):
    buy_conditions = data["ADR"] < data["price"]
    sell_conditions = data["ADR"] > data["price"]
    return buy_conditions, sell_conditions


def check_adx(data):
    buy_conditions = data["ADX"] > 20
    sell_conditions = data["ADX"] < 20
    return buy_conditions, sell_conditions


def check_mfi(data, tolerance: float):
    buy_threshold = 20.0
    sell_threshold = 80.0

    buy_conditions = data["MFI"] < (buy_threshold + tolerance)
    sell_conditions = data["MFI"] > (sell_threshold - tolerance)

    return buy_conditions, sell_conditions


def check_trix(data):
    buy_conditions = data["TRIX"] > data["TRIX_signal"]
    sell_conditions = data["TRIX"] < data["TRIX_signal"]
    return buy_conditions, sell_conditions


def check_tsi(data):
    buy_conditions = (data["TSI"] > data["TSI_signal"]) & (
        data["TSI"].shift(1) <= data["TSI_signal"].shift(1)
    )
    sell_conditions = (data["TSI"] < data["TSI_signal"]) & (
        data["TSI"].shift(1) >= data["TSI_signal"].shift(1)
    )
    tsi_oversold = data["TSI"] < -25
    tsi_overbought = data["TSI"] > 25
    return tsi_oversold, tsi_overbought


def check_williams_r(data):
    buy_conditions = data["Williams %R"] < -80
    sell_conditions = data["Williams %R"] > -20
    return buy_conditions, sell_conditions


def check_ultimate_oscillator(data):
    buy_conditions = data["Ultimate Oscillator"] < 30
    sell_conditions = data["Ultimate Oscillator"] > 70
    return buy_conditions, sell_conditions


def check_stochastic(data):
    buy_conditions = (data["Stoch_K"] < 20) & (data["Stoch_K"] > data["Stoch_D"])
    sell_conditions = (data["Stoch_K"] > 80) & (data["Stoch_K"] < data["Stoch_D"])
    return buy_conditions, sell_conditions


def detect_pullback(data):
    signals = pd.DataFrame(index=data.index)
    signals["RSI"] = ta.rsi(data["price"], length=14)
    bbands = ta.bbands(data["price"], length=20)
    signals["upper_band"] = bbands["BBU_20_2.0"]
    signals["lower_band"] = bbands["BBL_20_2.0"]
    signals["price_above_upper_band"] = data["price"] > signals["upper_band"]
    macd = ta.macd(data["price"], fast=12, slow=26, signal=9)
    signals["MACD"] = macd["MACD_12_26_9"]
    signals["MACD_Signal"] = macd["MACDs_12_26_9"]
    signals["MACD_Hist"] = macd["MACDh_12_26_9"]
    signals["MACD_Bearish_Cross"] = (signals["MACD"] < signals["MACD_Signal"]) & (
        signals["MACD"].shift(1) >= signals["MACD_Signal"].shift(1)
    )
    signals["Volume"] = data["volume"]

    signals["Potential_Pullback"] = (
        (signals["RSI"] > 70)
        | (signals["price_above_upper_band"])
        | (signals["MACD_Bearish_Cross"])
        | (
            (data["price"] > data["price"].rolling(window=50).mean())
            & (data["price"] > data["price"].rolling(window=200).mean())
        )
    )

    return signals


def perform_analysis(data=None, from_json=False, back_test=False):
    global price_data, asset_id
    with analysis_lock:
        logging.info("Performing technical analysis and generating signals...")

        if from_json:
            if data is None:
                logging.error("Data for analysis is not provided.")
                return
        else:
            data = price_data

        data = calculate_indicators(data)
        buy_signals, sell_signals, indicators = generate_signals(data)
        pullback_signals = detect_pullback(data)

        if not buy_signals.empty:
            logging.info(
                f"Buy signal detected at {buy_signals.index[-1].strftime('%Y-%m-%d %H:%M:%S')}. Might be old."
            )
            latest_buy_signal = buy_signals.iloc[-1]
            send_signal_to_alpaca_bot("buy", latest_buy_signal)
        if not sell_signals.empty:
            logging.info(
                f"Sell signal detected at {sell_signals.index[-1].strftime('%Y-%m-%d %H:%M:%S')}. Might be old."
            )
            latest_sell_signal = sell_signals.iloc[-1]
            send_signal_to_alpaca_bot("sell", latest_sell_signal)
        if not pullback_signals.empty:
            potential_pullback_times = pullback_signals[
                pullback_signals["Potential_Pullback"]
            ]
            if not potential_pullback_times.empty:
                logging.info(
                    f"Potential pullback detected at {potential_pullback_times.index[-1].strftime('%Y-%m-%d %H:%M:%S')}"
                )

        export_results(data, buy_signals, sell_signals, indicators)

        if back_test:
            profit_or_loss, trades_df = backtest_trading_signals(
                data, buy_signals, sell_signals
            )

            logging.info("\nTrading Actions:")
            logging.info(
                tabulate(trades_df, headers="keys", tablefmt="grid", showindex=False)
            )
            logging.info(
                f"Backtest result: {'✅ Profit' if profit_or_loss > 0 else '❌ Loss'} of ${abs(profit_or_loss):,.2f}"
            )

            return trades_df, profit_or_loss


def send_signal_to_coinbase_bot(signal_type, signal_data):
    url = "http://127.0.0.1:5000/coinbase-bot"
    payload = {
        "type": signal_type,
        "price": signal_data["price"],
        "timestamp": signal_data.name.strftime("%Y-%m-%d %H:%M:%S"),
    }
    response = requests.post(url, json=payload)
    logging.info(f"Response from trading bot: {response.text}")
    if response.status_code == 200:
        logging.info(f"✅ Successfully sent {signal_type} signal to trading bot.")
    else:
        logging.error(
            f"❌ Failed to send {signal_type} signal to trading bot. Error: {response.text}"
        )


def send_signal_to_alpaca_bot(signal_type, signal_data):
    url = "http://127.0.0.1:5000/alpaca-bot"
    payload = {
        "type": signal_type,
        "symbol": "BTC/USD",
        "price": signal_data["price"],
        "timestamp": signal_data.name.strftime("%Y-%m-%d %H:%M:%S"),
    }
    response = requests.post(url, json=payload)
    logging.info(f"Response from Alpaca bot: {response.text}")
    if response.status_code == 200:
        logging.info(f"✅ Successfully sent {signal_type} signal to Alpaca bot.")
    else:
        logging.error(
            f"❌ Failed to send {signal_type} signal to Alpaca bot. Error: {response.text}"
        )


def test_coinbase_trading_bot(signal_type, price):
    url = "http://127.0.0.1:5000/coinbase-bot"
    payload = {
        "type": signal_type,
        "price": price,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, data=json.dumps(payload), headers=headers)

    if response.status_code == 200:
        logging.info(f"Successfully sent {signal_type} signal to trading bot.")
    else:
        logging.error(
            f"Failed to send {signal_type} signal to trading bot. Status code: {response.status_code}"
        )
        logging.error(f"Response: {response.text}")


def export_results(data, buy_signals, sell_signals, indicators):
    result = {
        "symbol": get_clean_symbol(asset_id),
        "timestamp": data.index.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "price": data["price"].tolist(),
        "SMA": data["SMA"].tolist(),
        "EMA": data["EMA"].tolist(),
        "RSI": data["RSI"].tolist(),
        "upper_band": data["upper_band"].tolist(),
        "middle_band": data["middle_band"].tolist(),
        "lower_band": data["lower_band"].tolist(),
        "ATR": data["ATR"].tolist(),
        "Stoch_K": data["Stoch_K"].tolist(),
        "Stoch_D": data["Stoch_D"].tolist(),
        "OBV": data["OBV"].tolist(),
        "CCI": data["CCI"].tolist(),
        "VWAP": data["VWAP"].tolist(),
        "MACD": data["MACD"].tolist(),
        "MACD_Signal": data["MACD_Signal"].tolist(),
        "MACD_Hist": data["MACD_Hist"].tolist(),
        "ADX": data["ADX"].tolist(),
        "ADX+DI": data["ADX+DI"].tolist(),
        "ADX-DI": data["ADX-DI"].tolist(),
        "MFI": data["MFI"].tolist(),
        "TRIX": data["TRIX"].tolist(),
        "TRIX_signal": data["TRIX_signal"].tolist(),
        "TSI": data["TSI"].tolist(),
        "TSI_signal": data["TSI_signal"].tolist(),
        "Williams %R": data["Williams %R"].tolist(),
        "Ultimate Oscillator": data["Ultimate Oscillator"].tolist(),
        "buy_signals": buy_signals.index.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "buy_prices": buy_signals["price"].tolist(),
        "sell_signals": sell_signals.index.strftime("%Y-%m-%d %H:%M:%S").tolist(),
        "sell_prices": sell_signals["price"].tolist(),
        "indicators": indicators,
        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    output_dir = "../data/generated"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{asset_id}-ta-trading-signals.json")

    with open(output_file, "w") as f:
        json.dump(result, f, indent=4)

    # Compress using Gzip (typically 4 times smaller)
    gzip_file = output_file + ".gz"
    with open(output_file, "rb") as f_in, gzip.open(gzip_file, "wb") as f_out:
        data = f_in.read()
        f_out.write(data)
    logging.info(f"Gzip compressed JSON file saved as {gzip_file}")

    # Upload to S3. We can upload as often as we want since the file is compressed AND uploading costs are low (compared
    # to downloading)
    s3_key = f"trading/{asset_id}-ta-trading-signals.json.gz"
    upload_to_s3(gzip_file, s3_key, "../data/src/config.json")


def perform_one_time_analysis(
    asset_id, back_test=False, start_date=None, timespan=None
):
    if start_date is None:
        start_date = datetime.now().strftime("%Y-%m-%d")

    if timespan is None:
        timespan = "minute"

    fetch_data(asset_id, start_date, "1", timespan)

    asset_id = get_clean_symbol(asset_id)
    search_pattern = f"../data/src/{asset_id}-*.json"
    matching_files = glob.glob(search_pattern)

    if not matching_files:
        logging.error(f"No JSON file found for asset ID: {asset_id}")
        return

    json_file_path = matching_files[0]

    try:
        with open(json_file_path) as f:
            data_json = json.load(f)
    except FileNotFoundError:
        logging.error(f"File {json_file_path} not found.")
        return

    prices = data_json["data"][asset_id]["prices"]
    volumes = data_json["data"][asset_id]["volumes"]
    timestamps = [
        pd.to_datetime(price[0], unit="ms").tz_localize("UTC").tz_convert("US/Eastern")
        for price in prices
    ]
    prices = [price[1] for price in prices]
    volumes = [volume[1] for volume in volumes]

    data = pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": prices,
            "volume": volumes,
        }
    )

    for col in [
        "SMA",
        "EMA",
        "RSI",
        "upper_band",
        "middle_band",
        "lower_band",
        "ATR",
        "Stoch_K",
        "Stoch_D",
        "OBV",
        "CCI",
        "VWAP",
    ]:
        data[col] = None

    data.set_index("timestamp", inplace=True)
    perform_analysis(data, from_json=True, back_test=back_test)


def get_clean_symbol(ticker: str) -> str:
    """
    Clean the ticker ID by removing the 'x:' prefix and 'usd' suffix if present.

    :param ticker: The original ticker ID.
    :return: The cleaned ticker ID.
    """
    if ticker.startswith("x:"):
        ticker = ticker[2:]
    if ticker.endswith("usd"):
        ticker = ticker[:-3]
    return ticker


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
        bucket_name = credentials["bucket_name_public"]
        s3.upload_file(file_path, bucket_name, s3_key, ExtraArgs={"ACL": "public-read"})
        logging.info(
            f"File {file_path} uploaded to S3 bucket {bucket_name}. Key: {s3_key}"
        )
    except Exception as e:
        logging.error(f"Error uploading file to S3: {e}")


def backtest_trading_signals(data, buy_signals, sell_signals, initial_capital=50000):
    capital = initial_capital
    position = 0
    signals = []

    # Combine buy and sell signals into one DataFrame and sort by index
    buy_signals["signal"] = "buy"
    sell_signals["signal"] = "sell"
    combined_signals = pd.concat([buy_signals, sell_signals]).sort_index()

    # Filter out consecutive signals of the same type
    last_signal_type = None
    for idx, row in combined_signals.iterrows():
        if row["signal"] != last_signal_type:
            signals.append(row)
            last_signal_type = row["signal"]

    trades = []
    trade_number = 1
    for i, signal in enumerate(signals):
        if signal["signal"] == "buy":
            # Buy as many shares as possible with the available capital
            shares_bought = capital // signal["price"]
            capital -= shares_bought * signal["price"]
            position += shares_bought
            trades.append(
                [trade_number, "Buy", signal.name, signal["price"], shares_bought, 0.0]
            )
            trade_number += 1
        elif signal["signal"] == "sell" and position > 0:
            # Sell all the shares held
            profit_or_loss_trade = (signal["price"] - trades[-1][3]) * position
            capital += position * signal["price"]
            trades.append(
                [
                    trade_number,
                    "Sell",
                    signal.name,
                    signal["price"],
                    position,
                    profit_or_loss_trade,
                ]
            )
            position = 0  # Number of shares held
            trade_number += 1

    # Calculate final capital by selling any remaining position at the last price in data
    if position > 0:
        last_price = data["price"].iloc[-1]
        profit_or_loss_trade = (last_price - trades[-1][3]) * position
        capital += position * last_price
        trades.append(
            [
                trade_number,
                "Sell",
                data.index[-1],
                last_price,
                position,
                profit_or_loss_trade,
            ]
        )
        position = 0

    trades_df = pd.DataFrame(
        trades,
        columns=["Trade #", "Action", "Timestamp", "Price", "Shares", "Profit/Loss"],
    )
    profit_or_loss = capital - initial_capital

    return profit_or_loss, trades_df


def test_alpaca_trading_bot(limit_price: float):
    """
    Test the Alpaca trading bot by sending sample buy and sell signals. Make sure the bot is running locally.

    :return: None
    """
    logging.info("Testing Alpaca trading bot...")
    url = "http://127.0.0.1:5000/alpaca-bot"
    headers = {"Content-Type": "application/json"}

    signals = [
        {
            "type": "buy",
            "symbol": "BTC/USD",
            "price": limit_price,
            "timestamp": (datetime.utcnow() - timedelta(seconds=10)).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
        },
        # {
        #     "type": "sell",
        #     "symbol": "GME",
        #     "price": 26.12,
        #     "timestamp": (datetime.utcnow() - timedelta(seconds=5)).strftime(
        #         "%Y-%m-%d %H:%M:%S"
        #     ),
        # },
    ]

    for signal in signals:
        response = requests.post(url, json=signal, headers=headers)
        logging.info(f"Response from Alpaca bot: {response.text}")
        if response.status_code == 200:
            logging.info(f"✅ Successfully sent {signal['type']} signal to Alpaca bot.")
        else:
            logging.error(
                f"❌ Failed to send {signal['type']} signal to Alpaca bot. Error: {response.text}"
            )


def main():
    global asset_id
    parser = argparse.ArgumentParser(
        description="Stream crypto/stock data, perform TA, and send signals to trading bot(s)"
    )
    parser.add_argument("-c", "--coin", type=str, required=True, help="Coin/stock ID")
    parser.add_argument("--stream", action="store_true", help="Enable streaming mode")
    parser.add_argument(
        "-i",
        "--interval",
        type=str,
        choices=["second", "minute"],
        help="Stream interval (second or minute)",
    )
    parser.add_argument("--back-test", action="store_true", help="Enable backtest mode")
    parser.add_argument(
        "--start-date", type=str, help="Start date for fetching data (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--timespan",
        type=str,
        choices=["hour", "minute", "second"],
        help="Timespan for fetching data",
    )
    args = parser.parse_args()
    asset_id = args.coin

    if args.stream:
        if not args.interval:
            parser.error("--interval is required when --stream is specified")
        stream_price(args.interval)
    else:
        while True:
            perform_one_time_analysis(
                asset_id,
                back_test=args.back_test,
                start_date=args.start_date,
                timespan=args.timespan,
            )
            sleep_time = 5
            logging.info(f"Sleeping for {sleep_time} seconds...")
            time.sleep(sleep_time)


if __name__ == "__main__":
    main()

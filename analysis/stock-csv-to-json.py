"""
This script converts the stock data from a CSV file to a JSON format that can be used by the other Python scripts.
"""

import argparse
import json

import pandas as pd


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Convert stock data from CSV to JSON format."
    )
    parser.add_argument(
        "--symbol", type=str, required=True, help="Stock symbol for the data conversion"
    )
    parser.add_argument(
        "--category",
        type=str,
        required=True,
        choices=["open", "high", "low", "close"],
        help="Category of stock data to convert (open, high, low, close)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=False,
        help="Start date for the data (format YYYY-MM-DD)",
    )
    return parser.parse_args()


def convert_csv_to_json(symbol, category, filepath, output_path, start_date=None):
    df = pd.read_csv(filepath)

    df["Date"] = pd.to_datetime(df["Date"])

    if start_date:
        df = df[df["Date"] >= pd.to_datetime(start_date)]

    df["Timestamp"] = df["Date"].apply(lambda x: int(x.timestamp() * 1000))

    data_json = {
        "data": {
            symbol: {
                "prices": df[["Timestamp", category.capitalize()]].values.tolist(),
                "total_volumes": df[["Timestamp", "Volume"]].values.tolist(),
            }
        }
    }

    with open(output_path, "w") as f:
        json.dump(data_json, f, indent=4)


def main():
    args = parse_arguments()
    stock_symbol = args.symbol.lower()
    category = args.category.lower()

    # Generate file paths based on the stock symbol
    # Even though this is "generated" content, it is considered source for other scripts, hence the `src` directory
    file_path = f"../data/src/{stock_symbol.upper()}.csv"
    if args.start_date:
        output_json_path = (
            f"../data/src/{stock_symbol}-{category}-{args.start_date}.json"
        )
    else:
        output_json_path = f"../data/src/{stock_symbol}-{category}.json"

    convert_csv_to_json(
        stock_symbol, category, file_path, output_json_path, args.start_date
    )


if __name__ == "__main__":
    main()

import argparse
import os
import subprocess
import json

DAILY_DATA_FILE = "../data/src/coingecko-market_chart-data-daily.json"
COIN_IDS_FILE = "../data/src/coingecko-coin-ids-daily.json"


def main():
    parser = argparse.ArgumentParser(description="Process coin data")
    parser.add_argument(
        "-c", "--coin", type=str, required=True, help="Coin parameter for the script"
    )
    args = parser.parse_args()

    if os.path.exists(COIN_IDS_FILE):
        os.remove(COIN_IDS_FILE)
        print(f"{COIN_IDS_FILE} has been deleted.")
    else:
        print(f"{COIN_IDS_FILE} does not exist.")

    coin_data = {"ids": [args.coin]}

    with open(COIN_IDS_FILE, "w") as f:
        json.dump(coin_data, f)
    print(f"{COIN_IDS_FILE} has been created with coin ID: {args.coin}")

    if os.path.exists(DAILY_DATA_FILE):
        os.remove(DAILY_DATA_FILE)
        print(f"{DAILY_DATA_FILE} has been deleted.")

    fetch_command = ["python3", "fetch-coin-data.py", "-i", "daily"]
    subprocess.run(fetch_command, check=True)


if __name__ == "__main__":
    main()

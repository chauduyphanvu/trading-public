import argparse
import os
import subprocess
import json
from datetime import datetime, timedelta

HOURLY_DATA_FILE = "../data/src/coingecko-market_chart-data-hourly.json"
COIN_IDS_FILE = "../data/src/coingecko-coin-ids-hourly.json"


def get_week_day_and_index(date):
    week_day = date.strftime("%a")
    week_of_month = (date.day - 1) // 7 + 1
    return week_of_month, week_day


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

    if os.path.exists(HOURLY_DATA_FILE):
        os.remove(HOURLY_DATA_FILE)
        print(f"{HOURLY_DATA_FILE} has been deleted.")

    fetch_command = ["python3", "fetch-coin-data.py", "-i", "hourly"]
    subprocess.run(fetch_command, check=True)
    print(
        f"{HOURLY_DATA_FILE} has been created with hourly data for {args.coin} ✅✅✅"
    )

    today = datetime.today()

    for i in range(7):
        current_date = today + timedelta(days=i)
        week_index, week_day = get_week_day_and_index(current_date)

        swings_command = [
            "python3",
            "price-swings-hourly.py",
            str(week_index),
            week_day,
            "-c",
            args.coin,
            "--trend",
        ]
        print(
            f"Running command for {current_date.strftime('%Y-%m-%d')}: {' '.join(swings_command)}"
        )
        subprocess.run(swings_command, check=True)


if __name__ == "__main__":
    main()

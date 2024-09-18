import json
import time
from collections import OrderedDict
from datetime import datetime

import requests

API_URL = "https://api.solscan.io/v2/token/holders"
FETCH_INTERVAL_SECOND = 900  # 15 minutes
JSON_FILE_PATH = "../data/solscan-tokens.json"
OUTPUT_PATH_PREFIX = "../data/"
DEFAULT_OFFSET = "0"
DEFAULT_SIZE = "40"
API_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "identity",
    "Accept-Language": "en-US,en;q=0.9",
    "Dnt": "1",
    "Origin": "https://solscan.io",
    "Referer": "https://solscan.io/",
    "Sec-Ch-Ua": '"Microsoft Edge";v="123", "Not:A-Brand";v="8", "Chromium";v="123"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
    "Sec-Gpc": "1",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
    "Sol-Aut": "Ed=cXR4RVylTA4R4B9dls0fKtakecAuOmIx-8k0E",
}


def read_existing_data(file_name):
    try:
        with open(file_name, "r") as file:
            return json.load(file, object_pairs_hook=OrderedDict)
    except FileNotFoundError:
        return OrderedDict()


def write_data_to_json(file_name, ordered_keys, new_data):
    existing_data = read_existing_data(file_name)
    ordered_data = OrderedDict()

    # Add new entries and update existing ones in the ranked order of Solscan's returned list
    for key in ordered_keys:
        if key in existing_data:
            # If the holder exists, update their data
            existing_data[key].update(new_data.get(key, {}))
        else:
            # If it's a new holder, add them
            existing_data[key] = new_data.get(key)

        ordered_data[key] = existing_data[key]

    # Save the ordered data
    with open(file_name, "w") as file:
        json.dump(ordered_data, file, indent=4)


def ping_endpoint(token, coin_id):
    url = API_URL
    response = requests.get(
        url,
        params={"token": token, "offset": DEFAULT_OFFSET, "size": DEFAULT_SIZE},
        headers=API_HEADERS,
    )
    if response.status_code == 200:
        content = response.content
        content_json = json.loads(content.decode("utf-8"))
        data = content_json["data"]["result"]

        new_data = OrderedDict()
        now = datetime.utcnow().isoformat()
        ordered_keys = []
        for entry in data:
            owner = entry["owner"]
            amount = entry["amount"] / 10 ** entry["decimals"]  # Adjust for decimals
            amount_str = f"{amount:,.2f}"
            new_data[owner] = {now: amount_str}
            ordered_keys.append(owner)

        write_data_to_json(
            f"../data/{coin_id}_holders_data.json", ordered_keys, new_data
        )
        print(
            f"Data fetched and updated for {coin_id} at",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )
    else:
        print(
            f"Failed to fetch data for {coin_id} at",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        )


def main():
    with open(JSON_FILE_PATH, "r") as file:
        tokens = json.load(file)

    while True:
        for token in tokens:
            ping_endpoint(token["token"], token["coin_id"])
        time.sleep(FETCH_INTERVAL_SECOND)


if __name__ == "__main__":
    main()

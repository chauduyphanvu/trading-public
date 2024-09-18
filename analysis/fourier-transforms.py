import argparse
import json
import logging
from datetime import datetime, timedelta

import numpy as np
from scipy.fft import fft
import matplotlib.pyplot as plt

thresholds = {
    "bitcoin": 12000000,
    "dogecoin": 40,
    "myro": 0.02,
    "dogwifcoin": 20,
    "solana": 15000,
    "shiba-inu": 0.003,
    "ethereum": 100000,
    "gme": 0.002,
}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Perform FFT on cryptocurrency price data and analyze cycles."
    )
    parser.add_argument(
        "-c", "--base-coin", type=str, help="Base coin ID", required=True
    )
    parser.add_argument(
        "-f",
        "--input-file",
        type=str,
        help="Path to the price input JSON file",
        required=True,
    )
    return parser.parse_args()


def load_price_data(filename, coin_id):
    logging.info(f"Loading price data for coin ID '{coin_id}' from {filename}...")
    with open(filename, "r") as file:
        data = json.load(file)
        price_data = data["data"][coin_id]["prices"]
        return [item[1] for item in price_data], price_data[0][0]


def perform_fft(prices):
    price_fft = fft(prices)
    logging.info(
        "FFT performed on price data. Extracting frequencies, magnitudes, and phases..."
    )

    frequencies = np.fft.fftfreq(len(prices), d=1)[: len(prices) // 2]
    logging.info(f"Extracted {len(frequencies)} frequencies.")

    magnitudes = np.abs(price_fft)[: len(prices) // 2]
    logging.info(f"Extracted {len(magnitudes)} magnitudes.")

    phases = np.angle(price_fft)[: len(prices) // 2]
    logging.info(f"Extracted {len(phases)} phases.")

    return frequencies, magnitudes, phases


def analyze_cycles(frequencies, magnitudes, phases, first_date, today, threshold):
    grouped_cycles = []
    for freq, mag, phase in zip(frequencies, magnitudes, phases):
        if mag > threshold and freq > 0:
            logging.info(
                f"Found significant cycle. Magnitude {mag:.4f} > Threshold {threshold}. Frequency: {freq:.4f} > 0."
            )
            logging.info("Analyzing significant cycle...")

            period = 1 / freq
            start_time = (phase / (2 * np.pi * freq)) % period
            next_start_date = first_date + timedelta(days=start_time)

            while next_start_date <= today:
                conclusion_date = next_start_date + timedelta(days=period)
                cycle_entry = {
                    "frequency": freq,
                    "magnitude": mag,
                    "phase": phase,
                    "period": period,
                    "start_time": start_time,
                    "start_date": next_start_date.strftime("%Y-%m-%d"),
                    "conclusion_date": conclusion_date.strftime("%Y-%m-%d"),
                }
                grouped_cycles.append(cycle_entry)
                next_start_date = conclusion_date

    grouped_cycles.sort(key=lambda x: x["magnitude"], reverse=True)
    return grouped_cycles[:5]  # Return top 5 prominent cycles


def save_results(filename, data):
    with open(filename, "w") as f:
        json.dump({"cycles_by_frequency": data}, f, indent=4)


def plot_cycles(prices, cycles, base_coin, first_date):
    num_cycles = len(cycles)
    fig, axes = plt.subplots(num_cycles, 1, figsize=(12, 6 * num_cycles), sharex=True)

    if num_cycles == 1:
        axes = [axes]

    colors = plt.cm.viridis(np.linspace(0, 1, num_cycles))

    for idx, cycle in enumerate(cycles):
        ax = axes[idx]
        ax.plot(prices, label="Price Data")
        start_date = datetime.strptime(cycle["start_date"], "%Y-%m-%d")
        conclusion_date = datetime.strptime(cycle["conclusion_date"], "%Y-%m-%d")
        start_idx = (start_date - first_date).days
        end_idx = (conclusion_date - first_date).days

        while start_idx < len(prices):
            ax.axvspan(
                start_idx,
                end_idx,
                color=colors[idx],
                alpha=0.3,
                label=(
                    f"Cycle {cycle['frequency']:.4f} Hz"
                    if start_idx == (start_date - first_date).days
                    else ""
                ),
            )
            start_idx += int(cycle["period"])
            end_idx += int(cycle["period"])

        ax.set_ylabel("Price")
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


def main():
    args = parse_arguments()
    prices, first_date_timestamp = load_price_data(args.input_file, args.base_coin)
    first_date = datetime.utcfromtimestamp(first_date_timestamp / 1000.0)
    today = datetime.utcnow()
    frequencies, magnitudes, phases = perform_fft(prices)
    threshold = thresholds.get(
        args.base_coin, 1
    )  # Default threshold if coin ID is not found
    grouped_cycles = analyze_cycles(
        frequencies, magnitudes, phases, first_date, today, threshold
    )

    output_file = f"../data/generated/{args.base_coin}-fft-data.json"
    save_results(output_file, grouped_cycles)

    logging.info(
        f"Fourier Transform data has been saved to {output_file} for coin ID '{args.base_coin}'."
    )
    logging.info(
        "Logging significant cycles grouped by frequency and ranked by magnitude:"
    )
    # for cycle in grouped_cycles:
    #     logging.info(
    #         f"Frequency: {cycle['frequency']:.4f}, "
    #         f"Period: {cycle['period']:.2f} days, "
    #         f"Start Date: {cycle['start_date']}, "
    #         f"Conclusion Date: {cycle['conclusion_date']}"
    #     )

    # for freq, cycles in grouped_cycles.items():
    #     logging.info(f"Frequency: {freq:.4f}")
    #     for cycle in cycles:
    #         logging.info(
    #             f"\tPeriod: {cycle['period']:.2f} days, "
    #             f"Start Date: {cycle['start_date']}, "
    #             f"Conclusion Date: {cycle['conclusion_date']}"
    #         )

    # Plot the cycles
    plot_cycles(prices, grouped_cycles, args.base_coin, first_date)


if __name__ == "__main__":
    main()

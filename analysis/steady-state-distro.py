import argparse
import json
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import linregress
from sklearn.cluster import KMeans

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def get_time_unit_from_filename(file_path):
    file_name = os.path.basename(file_path)
    if "minute" in file_name.lower():
        return "Minutes"
    elif "hour" in file_name.lower():
        return "Hours"
    elif "day" in file_name.lower():
        return "Days"
    else:
        return "Time Units"


def steady_state_distribution(transition_matrix):
    eigvals, eigvecs = np.linalg.eig(transition_matrix.T)
    steady_state = np.real(eigvecs[:, np.isclose(eigvals, 1)])
    steady_state = steady_state / steady_state.sum()
    return steady_state


def load_price_data(file_path, coin_id):
    with open(file_path, "r") as file:
        data = json.load(file)
    return [price[1] for price in data["data"][coin_id]["prices"]]


def define_bins_kmeans(prices, num_bins=5):
    prices = np.array(prices).reshape(-1, 1)
    kmeans = KMeans(n_clusters=num_bins, random_state=0).fit(prices)
    bin_edges = np.sort(kmeans.cluster_centers_.flatten())
    bin_edges = np.concatenate(([prices.min()], bin_edges, [prices.max()]))
    bin_edges = np.unique(bin_edges)  # Ensure bin edges are unique
    labels = range(len(bin_edges) - 1)
    return bin_edges, labels


def define_bins(prices, num_bins=5, method="quantile"):
    if method == "equal":
        bins = np.linspace(min(prices), max(prices), num=num_bins)
    elif method == "quantile":
        bins = np.quantile(prices, np.linspace(0, 1, num_bins))
    labels = range(len(bins) - 1)
    return bins, labels


def create_transition_matrix(price_bins):
    price_bins_df = pd.DataFrame({"price_bins": price_bins})
    transition_matrix = pd.crosstab(
        price_bins_df.price_bins.shift(1), price_bins_df.price_bins, normalize="index"
    ).fillna(0)
    return transition_matrix


def forecast_prices(transition_matrix, initial_bin, steps):
    current_bin = initial_bin
    future_prices = []

    for _ in range(steps):
        current_bin = np.random.choice(
            transition_matrix.shape[1], p=transition_matrix[current_bin]
        )
        future_prices.append(current_bin)

    return future_prices


def plot_steady_state_distribution(steady_state, bins, labels, coin_id, save_path=None):
    steady_state_probs = steady_state.flatten()
    plt.figure(figsize=(10, 6))
    plt.bar(
        labels,
        steady_state_probs * 100,
        tick_label=[f"Bin {i}\n({bins[i]:.6f} --> {bins[i + 1]:.6f})" for i in labels],
    )
    plt.title(
        f"{coin_id} | Steady-State Distribution"
        f"\nThis graph shows the percentage of time spent in each price range. It is calculated using the steady-state "
        f"distribution of the price transition matrix. The price transition matrix is created by binning the prices "
        f"into {len(bins) - 1} price ranges and calculating the probability of moving from one price range to another."
        f"To use this information, you can identify the price ranges where the coin spends the most time and use them "
        f"as entry and exit points for trading strategies."
    )
    plt.xlabel("Bins (Price Ranges)")
    plt.ylabel("Percentage of Time (%)")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)


def modify_transition_matrix_for_absorption(transition_matrix, target_bin):
    n = transition_matrix.shape[0]
    if target_bin >= n:
        raise ValueError(
            f"target_bin {target_bin} is out of bounds for transition matrix with size {n}"
        )
    modified_matrix = transition_matrix.copy()
    modified_matrix[target_bin] = np.eye(n)[target_bin]  # Make the target bin absorbing
    return modified_matrix


def expected_steps_to_reach_bin(transition_matrix, target_bin):
    n = transition_matrix.shape[0]
    if target_bin >= n:
        raise ValueError(
            f"target_bin {target_bin} is out of bounds for transition matrix with size {n}"
        )
    modified_matrix = modify_transition_matrix_for_absorption(
        transition_matrix, target_bin
    )

    logging.info(
        f"Modified transition matrix for target bin {target_bin}:\n{modified_matrix}"
    )

    Q = np.delete(np.delete(modified_matrix, target_bin, axis=0), target_bin, axis=1)

    I = np.eye(Q.shape[0])
    N = np.linalg.inv(I - Q)

    expected_steps = N.sum(axis=1)
    expected_steps_with_target = np.insert(expected_steps, target_bin, 1)

    return expected_steps_with_target


def plot_expected_steps_heatmap(expected_steps, labels, coin_id, save_path=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        expected_steps,
        annot=True,
        fmt="g",
        cmap="viridis",
        cbar=True,
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(f"Expected Number of Steps to Reach Each Bin for {coin_id}")
    plt.xlabel("Target Bin")
    plt.ylabel("Starting Bin")

    if save_path:
        plt.savefig(save_path)


def plot_prices_with_bins(
    prices,
    bins,
    steady_state_probs,
    coin_id,
    method,
    optimal_entry_bin=None,
    optimal_exit_bin=None,
    save_path=None,
):
    plt.figure(figsize=(14, 8))
    plt.plot(prices, label=f"{coin_id} Prices")

    for i in range(len(steady_state_probs)):
        plt.axhspan(
            bins[i],
            bins[i + 1],
            alpha=0.3,
            color=f"C{i}",
            label=f"Bin {i} ({bins[i]:.6f} --> {bins[i + 1]:.6f}), {steady_state_probs[i] * 100:.2f}% of time",
        )

        mid_point = (bins[i] + bins[i + 1]) / 2
        plt.text(
            len(prices) + 50 / 2,
            mid_point,
            f"{steady_state_probs[i] * 100:.2f}% of time",
            horizontalalignment="center",
            verticalalignment="center",
            bbox=dict(facecolor="white", alpha=0.5),
        )

    if optimal_entry_bin is not None and optimal_exit_bin is not None:
        plt.axhspan(
            bins[optimal_entry_bin],
            bins[optimal_entry_bin + 1],
            color="green",
            alpha=0.5,
            label="Optimal Entry Point",
        )
        plt.axhspan(
            bins[optimal_exit_bin],
            bins[optimal_exit_bin + 1],
            color="red",
            alpha=0.5,
            label="Optimal Exit Point",
        )

    plt.title(
        f"{coin_id} | Prices with {method.capitalize()} Bins\n"
        f"Percentage of time spent in each price range based on steady-state distribution of price transition "
        f"matrix"
    )
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend(loc="upper left")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)


def calculate_up_down_probabilities(transition_matrix):
    n = transition_matrix.shape[0]
    move_up_probabilities = np.zeros(n)
    move_down_probabilities = np.zeros(n)

    for i in range(n):
        if i > 0:
            move_down_probabilities[i] = np.sum(transition_matrix[i, :i])
        if i < n - 1:
            move_up_probabilities[i] = np.sum(transition_matrix[i, i + 1 :])

    return move_up_probabilities, move_down_probabilities


def plot_up_down_probabilities(
    move_up_probabilities, move_down_probabilities, labels, coin_id, save_path=None
):
    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(labels))

    plt.bar(index, move_up_probabilities, bar_width, label="Up")
    plt.bar(index + bar_width, move_down_probabilities, bar_width, label="Down")

    plt.xlabel("Bins")
    plt.ylabel("Probability")
    plt.title(f"{coin_id} | Probabilities of Moving Up and Down")
    plt.xticks(index + bar_width / 2, labels)
    plt.legend()

    if save_path:
        plt.savefig(save_path)


def calculate_expected_returns(transition_matrix, bins):
    n = transition_matrix.shape[0]
    expected_returns = np.zeros(n)

    for i in range(n):
        bin_midpoints = (bins[i] + bins[i + 1]) / 2
        for j in range(n):
            next_bin_midpoint = (bins[j] + bins[j + 1]) / 2
            expected_returns[i] += transition_matrix[i, j] * (
                next_bin_midpoint - bin_midpoints
            )

    return expected_returns


def plot_expected_returns(expected_returns, bins, coin_id, save_path=None):
    plt.figure(figsize=(10, 6))
    labels = [
        f"Bin {i}\n({bins[i]:.6f} --> {bins[i + 1]:.6f})" for i in range(len(bins) - 1)
    ]
    plt.bar(labels, expected_returns)
    plt.title(f"{coin_id} | Expected Returns for Each Bin")
    plt.xlabel("Bins (Price Ranges)")
    plt.ylabel("Expected Return")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)


def analyze_regime_characteristics(prices, regimes):
    regime_stats = {}
    unique_regimes = np.unique(regimes)

    for regime in unique_regimes:
        regime_indices = np.where(regimes == regime)[0]
        regime_prices = np.array(prices)[regime_indices]

        mean_price = np.mean(regime_prices)
        std_price = np.std(regime_prices)

        slope, intercept, r_value, p_value, std_err = linregress(
            range(len(regime_prices)), regime_prices
        )

        regime_stats[regime] = {
            "mean_price": mean_price,
            "std_price": std_price,
            "slope": slope,
            "intercept": intercept,
            "r_value": r_value,
            "p_value": p_value,
            "std_err": std_err,
        }

    return regime_stats


def print_regime_characteristics(regime_stats):
    for regime, stats in regime_stats.items():
        mean_price = stats["mean_price"]
        std_price = stats["std_price"]

        one_std_lower = mean_price - std_price
        one_std_upper = mean_price + std_price
        two_std_lower = mean_price - 2 * std_price
        two_std_upper = mean_price + 2 * std_price

        logging.info(f"Regime {regime}:")
        logging.info(f"  Mean Price: {mean_price:.6f}")
        logging.info(f"  Std Dev of Price: {std_price:.6f}")
        logging.info(f"  Trend (slope): {stats['slope']:.6f}")
        logging.info(f"  Intercept: {stats['intercept']:.6f}")
        logging.info(f"  R-squared: {stats['r_value'] ** 2:.6f}")
        logging.info(f"  P-value: {stats['p_value']:.6f}")
        logging.info(f"  Standard Error: {stats['std_err']:.6f}")

        logging.info(f"  1 Std Dev Range: {one_std_lower:.6f} to {one_std_upper:.6f}")
        logging.info(f"  2 Std Dev Range: {two_std_lower:.6f} to {two_std_upper:.6f}")
        logging.info("")


def identify_market_regimes(prices, num_regimes=3):
    prices = np.array(prices).reshape(-1, 1)
    kmeans = KMeans(n_clusters=num_regimes, random_state=0).fit(prices)
    regimes = kmeans.predict(prices)
    return regimes


def calculate_regime_transition_matrix(regimes):
    regime_series = pd.Series(regimes)
    transition_matrix = pd.crosstab(
        regime_series.shift(1), regime_series, normalize="index"
    ).fillna(0)
    return transition_matrix


def plot_market_regimes(
    prices, regimes, transition_matrix, coin_id, regime_stats, time_unit, save_path=None
):
    plt.figure(figsize=(14, 8))

    plt.plot(prices, color="gray", alpha=0.3, label=f"{coin_id} Prices")

    unique_regimes = np.unique(regimes)
    color_map = plt.cm.get_cmap("tab10", len(unique_regimes))

    poly_degree = 3

    for regime in unique_regimes:
        regime_indices = np.where(regimes == regime)[0]
        regime_segments = np.column_stack(
            [regime_indices, np.array(prices)[regime_indices]]
        )

        mean_price = regime_stats[regime]["mean_price"]
        std_price = regime_stats[regime]["std_price"]

        p = np.poly1d(
            np.polyfit(regime_indices, np.array(prices)[regime_indices], poly_degree)
        )
        trend_line = p(regime_indices)

        plt.plot(
            [],
            [],
            color=color_map(regime),
            label=f"Regime {regime}: Mean={mean_price:.6f}",
        )

        for i in range(len(regime_segments) - 1):
            if regime_indices[i + 1] - regime_indices[i] == 1:
                plt.plot(
                    regime_segments[i : i + 2, 0],
                    regime_segments[i : i + 2, 1],
                    color=color_map(regime),
                    linewidth=1,
                )

        plt.axhline(mean_price, color=color_map(regime), linestyle="--", linewidth=1.5)
        plt.text(
            len(prices) + 50,
            mean_price,
            f"${mean_price:.6f} (Avg Price)",
            color=color_map(regime),
            verticalalignment="center",
            horizontalalignment="left",
        )

        plt.fill_between(
            range(len(prices) + 50),
            mean_price - std_price,
            mean_price + std_price,
            color=color_map(regime),
            alpha=0.1,
            label=f"Regime {regime} 1 std dev",
        )
        plt.fill_between(
            range(len(prices) + 50),
            mean_price - 2 * std_price,
            mean_price + 2 * std_price,
            color=color_map(regime),
            alpha=0.05,
            label=f"Regime {regime} 2 std dev",
        )

        plt.text(
            len(prices) + 50,
            mean_price - std_price,
            f"${mean_price - std_price:.6f} (1 Std Dev)",
            color=color_map(regime),
            verticalalignment="bottom",
            horizontalalignment="left",
        )
        plt.text(
            len(prices) + 50,
            mean_price + std_price,
            f"${mean_price + std_price:.6f} (1 Std Dev)",
            color=color_map(regime),
            verticalalignment="top",
            horizontalalignment="left",
        )

        plt.text(
            len(prices) + 50,
            mean_price - 2 * std_price,
            f"${mean_price - 2 * std_price:.6f} (2 Std Dev)",
            color=color_map(regime),
            verticalalignment="bottom",
            horizontalalignment="left",
        )
        plt.text(
            len(prices) + 50,
            mean_price + 2 * std_price,
            f"${mean_price + 2 * std_price:.6f} (2 Std Dev)",
            color=color_map(regime),
            verticalalignment="top",
            horizontalalignment="left",
        )

        plt.plot(
            regime_indices,
            trend_line,
            color=color_map(regime),
            linestyle="-",
            linewidth=1,
            alpha=0.5,
            label=f"Regime {regime} Trend",
        )

    plt.title(
        f"{coin_id} | Market Regimes Analysis (via K-means Clustering)\n"
        "Regime Segments (Price Ranges), Mean Prices, Standard Deviation Bands, and Trend Lines"
    )
    plt.xlabel(f"Time ({time_unit})")
    plt.ylabel("Price (USD)")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)  # Save Market Regimes chart

    # Plotting Regime Transition Probabilities separately
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        transition_matrix,
        annot=True,
        cmap="viridis",
        cbar=True,
        xticklabels=[f"Regime {i}" for i in range(transition_matrix.shape[1])],
        yticklabels=[f"Regime {i}" for i in range(transition_matrix.shape[0])],
    )
    plt.title(f"{coin_id} | Regime Transition Probabilities")
    plt.xlabel("Next Regime")
    plt.ylabel("Current Regime")


def main(args):
    # Ensure the directory exists
    output_dir = "../data/generated/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    prices = load_price_data(args.file, args.coin)

    kmeans_bins, kmeans_labels = define_bins_kmeans(prices, num_bins=5)

    kmeans_price_bins = pd.cut(
        prices,
        bins=kmeans_bins,
        labels=kmeans_labels,
        include_lowest=True,
        duplicates="drop",
    )

    kmeans_transition_matrix = create_transition_matrix(kmeans_price_bins)

    kmeans_steady_state = steady_state_distribution(kmeans_transition_matrix.values)

    save_path_steady_state = (
        f"{output_dir}{args.coin.upper()}-steady-state-distribution.png"
    )

    plot_prices_with_bins(
        prices,
        kmeans_bins,
        kmeans_steady_state.flatten(),
        args.coin.upper(),
        "K-means",
        save_path=save_path_steady_state,
    )

    kmeans_expected_steps_matrix = []
    for target_bin in kmeans_labels:
        if target_bin < len(kmeans_transition_matrix):
            kmeans_expected_steps = expected_steps_to_reach_bin(
                kmeans_transition_matrix.values, target_bin
            )
            kmeans_expected_steps_matrix.append(kmeans_expected_steps)
            logging.info(
                f"Expected number of steps to reach bin {target_bin} for {args.coin.upper()} (K-means): {kmeans_expected_steps}"
            )

    kmeans_expected_steps_matrix = np.array(kmeans_expected_steps_matrix)

    save_path_heatmap = f"{output_dir}{args.coin.upper()}-expected-steps-heatmap.png"

    if kmeans_expected_steps_matrix.shape[0] == kmeans_expected_steps_matrix.shape[1]:
        plot_expected_steps_heatmap(
            kmeans_expected_steps_matrix,
            kmeans_labels,
            f"{args.coin.upper()} (K-means)",
            save_path=save_path_heatmap,
        )

    move_up_probabilities, move_down_probabilities = calculate_up_down_probabilities(
        kmeans_transition_matrix.values
    )

    optimal_entry_bin = np.argmax(move_up_probabilities)
    optimal_exit_bin = np.argmax(move_down_probabilities)

    logging.info(
        f"Optimal entry point (highest probability of moving up): Bin {optimal_entry_bin} ({kmeans_bins[optimal_entry_bin]:.6f} --> {kmeans_bins[optimal_entry_bin + 1]:.6f})"
    )
    logging.info(
        f"Optimal exit point (highest probability of moving down): Bin {optimal_exit_bin} ({kmeans_bins[optimal_exit_bin]:.6f} --> {kmeans_bins[optimal_exit_bin + 1]:.6f})"
    )

    save_path_optimal_entry_exit = (
        f"{output_dir}{args.coin.upper()}-optimal-entry-exit.png"
    )

    plot_prices_with_bins(
        prices,
        kmeans_bins,
        kmeans_steady_state.flatten(),
        args.coin.upper(),
        "K-means",
        optimal_entry_bin=optimal_entry_bin,
        optimal_exit_bin=optimal_exit_bin,
        save_path=save_path_optimal_entry_exit,
    )

    regimes = identify_market_regimes(prices, num_regimes=3)

    regime_transition_matrix = calculate_regime_transition_matrix(regimes)

    regime_stats = analyze_regime_characteristics(prices, regimes)

    print_regime_characteristics(regime_stats)

    time_unit = get_time_unit_from_filename(args.file)

    save_path_market_regimes = f"{output_dir}{args.coin.upper()}-market-regimes.png"

    plot_market_regimes(
        prices,
        regimes,
        regime_transition_matrix,
        args.coin.upper(),
        regime_stats,
        time_unit,
        save_path=save_path_market_regimes,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process cryptocurrency price data.")
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        required=True,
        help="Path to the JSON file containing price data",
    )
    parser.add_argument(
        "-c", "--coin", type=str, required=True, help="Coin ID to analyze"
    )

    args = parser.parse_args()
    main(args)

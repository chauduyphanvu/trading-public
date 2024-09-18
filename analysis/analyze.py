"""
This script is a wrapper around the individual analysis scripts.
"""

import concurrent.futures
import subprocess
import sys

if len(sys.argv) < 2:
    print("Usage: python analyze.py -c <coingecko-coin-id> -f <filename>")
    sys.exit(1)

base_coin = sys.argv[2]
file_name = sys.argv[4]

# Run "price-vol.py" first because a subset of other scripts depend on its output
# subprocess.run(["python3", "price-vol.py", "--base-coin", base_coin])

# Append the script names to this list to run them concurrently
scripts = [
    # "correlation.py",
    # "volatility.py",
    "steady-state-distro.py",
    "kelly-criterion.py",
    "mcmc.py",
    # "fourier-transforms.py",
    "price-swings-daily.py",
    # "wavelet-transform.py",
    # "stl.py",
]


def run_script(script):
    print(f"Running {script} with base-coin {base_coin}...")
    subprocess.run(["python3", script, "-c", base_coin, "-f", file_name])
    print("\n")  # Print a blank line for readability after each script execution


# Use ThreadPoolExecutor to run scripts concurrently
with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(run_script, script) for script in scripts]
    concurrent.futures.wait(
        futures
    )  # Optional: wait for all to complete before exiting

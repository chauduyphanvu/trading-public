import pandas as pd

# Load the data into a DataFrame
file_path = "../data/src/GME-abridged.csv"  # Update this with the actual file path
data = pd.read_csv(file_path, parse_dates=["Date"])


# Function to calculate the best holding periods for a given number of weeks from a custom start date
def calculate_best_weeks(data, num_weeks, start_date):
    # Filter the data to start from the custom start date
    filtered_data = data[data["Date"] >= pd.to_datetime(start_date)].reset_index(
        drop=True
    )

    # Convert the number of weeks to the number of days
    num_days = num_weeks * 5  # Assuming a 5-day trading week
    max_hodl_period = len(filtered_data) - num_days
    best_period = None
    max_return = -float("inf")

    for start_index in range(max_hodl_period + 1):
        end_index = start_index + num_days - 1
        if end_index >= len(filtered_data):
            continue
        start_price = filtered_data.iloc[start_index]["Close"]
        end_price = filtered_data.iloc[end_index]["Close"]
        hodl_return = (end_price - start_price) / start_price

        if hodl_return > max_return:
            max_return = hodl_return
            best_period = (start_index, end_index, hodl_return)

    return best_period, filtered_data


# Number of weeks to hold
num_weeks = 5

# Custom start date
start_date = "2024-01-10"

# Calculate the best holding period from the custom start date
best_period, filtered_data = calculate_best_weeks(data, num_weeks, start_date)

if best_period:
    start_index, end_index, best_return = best_period
    buy_date = filtered_data.iloc[start_index]["Date"]
    sell_date = filtered_data.iloc[end_index]["Date"]
    buy_price = filtered_data.iloc[start_index]["Close"]
    sell_price = filtered_data.iloc[end_index]["Close"]
    final_capital = 10000 * (1 + best_return)

    print(f"\nBest period to hold for {num_weeks} weeks starting from {start_date}:")
    print(f"Buy Date: {buy_date.strftime('%Y-%m-%d')}, Buy Price: ${buy_price:.2f}")
    print(f"Sell Date: {sell_date.strftime('%Y-%m-%d')}, Sell Price: ${sell_price:.2f}")
    print(f"Return: {best_return:.2f}x")
    print(f"Final Capital: ${final_capital:.2f}")
else:
    print(
        f"No valid period found for holding {num_weeks} weeks starting from {start_date}."
    )

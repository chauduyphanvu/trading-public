import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Read the JSON data from a local file
file_path = "../data/src/sol-1-hour-2024-07-05-to-2024-07-06.json"
with open(file_path, "r") as file:
    data = json.load(file)

# Extract price and volume data
timestamps, prices, volumes = [], [], []
for price_data, volume_data in zip(
    data["data"]["sol"]["prices"], data["data"]["sol"]["volumes"]
):
    timestamps.append(
        datetime.utcfromtimestamp(price_data[0] / 1000)
    )  # Convert to datetime
    prices.append(price_data[1])
    volumes.append(volume_data[1])

# Create a DataFrame for better handling
df = pd.DataFrame({"Timestamp": timestamps, "Price": prices, "Volume": volumes})

# Plotting the data
fig, ax1 = plt.subplots()

ax1.set_xlabel("Timestamp")
ax1.set_ylabel("Price", color="tab:blue")
ax1.plot(df["Timestamp"], df["Price"], color="tab:blue", label="Price")
ax1.tick_params(axis="y", labelcolor="tab:blue")

# ax2 = ax1.twinx()
# ax2.set_ylabel('Volume', color='tab:orange')
# ax2.plot(df['Timestamp'], df['Volume'], color='tab:orange', label='Volume')
# ax2.tick_params(axis='y', labelcolor='tab:orange')

fig.tight_layout()
plt.title("SOL Price and Volume Over Time")
plt.show()

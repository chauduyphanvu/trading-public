import pandas as pd
from pulp import LpMaximize, LpProblem, LpVariable, lpSum

# Load the CSV data
df = pd.read_csv("../data/src/GME-abridged.csv")

# Ensure the Date column is treated as a datetime object
df["Date"] = pd.to_datetime(df["Date"])

# Add a column for the day of the week
df["DayOfWeek"] = df["Date"].dt.day_name()

# Calculate technical indicators
# Moving Average (20 days)
df["MA20"] = df["Close"].rolling(window=20).mean()
# Relative Strength Index (14 days)
delta = df["Close"].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
df["RSI"] = 100 - (100 / (1 + rs))
# Moving Average Convergence Divergence (MACD)
df["MACD"] = (
    df["Close"].ewm(span=12, adjust=False).mean()
    - df["Close"].ewm(span=26, adjust=False).mean()
)
# Bollinger Bands
df["BB_upper"] = (
    df["Close"].rolling(window=20).mean() + 2 * df["Close"].rolling(window=20).std()
)
df["BB_lower"] = (
    df["Close"].rolling(window=20).mean() - 2 * df["Close"].rolling(window=20).std()
)

# Drop rows with NaN values in indicators
df.dropna(inplace=True)

# Create the ILP model
model = LpProblem(name="trading-strategy-multiple", sense=LpMaximize)

# Create binary variables for buy and sell decisions for each specific date
buy = {date: LpVariable(f"buy_{date}", cat="Binary") for date in df["Date"]}
sell = {date: LpVariable(f"sell_{date}", cat="Binary") for date in df["Date"]}

# Create interaction variables for buy-sell combinations
buy_sell = {
    (buy_date, sell_date): LpVariable(f"buy_{buy_date}_sell_{sell_date}", cat="Binary")
    for buy_date in df["Date"]
    for sell_date in df["Date"]
    if sell_date > buy_date
}

# Maximum number of trades
max_trades = 10

# Add constraints to limit the number of trades
model += lpSum(buy[date] for date in df["Date"]) <= max_trades
model += lpSum(sell[date] for date in df["Date"]) <= max_trades

# Add constraints to link interaction variables with buy and sell decisions
for buy_date in df["Date"]:
    for sell_date in df["Date"]:
        if sell_date > buy_date:
            model += buy_sell[(buy_date, sell_date)] <= buy[buy_date]
            model += buy_sell[(buy_date, sell_date)] <= sell[sell_date]

# Ensure exactly one buy and one sell in the week
model += lpSum(buy[date] for date in df["Date"]) <= max_trades
model += lpSum(sell[date] for date in df["Date"]) <= max_trades

# Ensure buys and sells are balanced
model += lpSum(buy[date] for date in df["Date"]) == lpSum(
    sell[date] for date in df["Date"]
)

# Objective: Maximize profit by choosing the best days to buy and sell
profit = lpSum(
    (
        df.loc[df["Date"] == sell_date, "Close"].values[0]
        - df.loc[df["Date"] == buy_date, "Open"].values[0]
    )
    * buy_sell[(buy_date, sell_date)]
    for buy_date in df["Date"]
    for sell_date in df["Date"]
    if sell_date > buy_date
)
model += profit

# Solve the ILP model
model.solve()

# Display the results
print("\nBuy decisions:")
buy_dates = [date for date in df["Date"] if buy[date].varValue > 0]
for date in buy_dates:
    print(f"{date}: Buy")

print("\nSell decisions:")
sell_dates = [date for date in df["Date"] if sell[date].varValue > 0]
for date in sell_dates:
    print(f"{date}: Sell")

# Calculate the total profit starting with $10,000
initial_capital = 10000
capital = initial_capital
transaction_cost_percentage = 0.001  # 0.1% transaction cost

# Print transaction details
print("\nTransaction details:")
for buy_date, sell_date in zip(buy_dates, sell_dates):
    buy_price = df.loc[df["Date"] == buy_date, "Open"].values[0]
    sell_price = df.loc[df["Date"] == sell_date, "Close"].values[0]

    # Calculate transaction costs
    transaction_costs = transaction_cost_percentage * (buy_price + sell_price)

    # Calculate the number of shares bought
    shares_bought = capital / buy_price

    # Update the capital
    capital = shares_bought * sell_price - transaction_costs

    print(
        f"Bought on {buy_date} at ${buy_price:.2f}, sold on {sell_date} at ${sell_price:.2f}"
    )
    print(
        f"Shares bought: {shares_bought:.2f}, Transaction costs: ${transaction_costs:.2f}"
    )
    print(f"Capital after transaction: ${capital:.2f}")

# Calculate the profit
profit = capital - initial_capital

print(f"\nFinal capital: ${capital:.2f}")
print(f"Profit: ${profit:.2f}")

# Display the optimal holding periods
for buy_date in buy_dates:
    for sell_date in sell_dates:
        if sell_date > buy_date:
            holding_time = (sell_date - buy_date).days
            print(
                f"\nOptimal holding time for buying on {buy_date} and selling on {sell_date}: {holding_time} days"
            )

import datetime as dt
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from solving_methods import *


degree = 5  # Degree of polynomial


def check_ticker_existence(symbol):
    try:
        ticker = yf.Ticker(symbol)
        history = ticker.history(period="1mo")
        if not history.empty:
            print(f"The ticker {symbol} is valid.")
            return True
        else:
            print(f"The ticker {symbol} is NOT valid.")
            return False
    except Exception as e:
        print(f"Exception arose while checking the existence of {symbol}: {e}")
        return False



s = input("Enter ticker of company: ")
s = s.upper()
while not check_ticker_existence(s):
    s = input("Enter ticker of company: ")
    s = s.upper()

stock = yf.Ticker(s)
available_interval_range = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]

interval = input("Enter interval. Please note that the full range of intervals available are: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo(enter in this format): ")
while interval.lower() not in available_interval_range:
    interval = input("Incorrect interval. Please note that the full range of intervals available are: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo(enter in this format): ")
interval = interval.lower()

while True:
    start_date_str = input("Enter the start date (YYYY-MM-DD): ")
    try:
        start_date = dt.datetime.strptime(start_date_str, "%Y-%m-%d")
        if start_date >= dt.datetime.today():
            print("Invalid date. ")
        else:
            print("Start date:", start_date)
            break
    except ValueError:
        print("Invalid date format. Please enter the date in YYYY-MM-DD format. Try again.")


stock_history = stock.history(start=start_date, end=dt.datetime.now(), interval=interval)
closing_prices = stock_history['Close'].to_frame()

# Getting Data finished. Turning into "equation" and solving starts now

b = np.matrix([round(closing_prices['Close'].iloc[i], 5) for i in range(len(closing_prices))]).T
map_of_t = pd.Series(map(lambda x: x / 100.0, range(len(stock_history.index)))).values
A = generate_A(range(len(map_of_t)), degree)  # Degree should be adequate
x = pseudo_inverse_CGS(A) @ b

# Plotting...
# plt.plot(map_of_t, b)
# plt.plot(map_of_t, [sum(x[i, 0] * (a ** i) for i in range(degree)) for a in map_of_t])
# plt.show()

# Now time for constrainted least squares

d1 = round(stock_history['Close'].to_frame()['Close'].iloc[0], 5)
d2 = round(stock_history['Close'].to_frame()['Close'].iloc[-1], 5)
d = np.array([[d1], [d2]])
C = generate_A([0, len(map_of_t)], degree)

# 2 * A.T @ A, C.T
# C, np.zeroes((2, 2))
KKT_Matrix = np.block([[ 2 * A.T @ A, C.T], [C, np.zeros((2, 2)).astype(np.float64)]])
KKT_Vector = np.block([[2 * A.T @ b], [d]])

x = pseudo_inverse_CGS(KKT_Matrix) @ KKT_Vector

plt.plot(map_of_t, b)
pred_x = [i / 10000.0 for i in range(len(map_of_t)*100)]
plt.plot(pred_x, [sum(x[i, 0] * (a ** i) for i in range(degree)) for a in pred_x])
plt.show()
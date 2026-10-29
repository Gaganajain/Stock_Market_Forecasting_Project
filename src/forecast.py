"""Stock Market Forecasting: compare ARIMA, Prophet and LSTM on a stock's closing price.

Usage:
    python forecast.py --ticker AAPL --start 2015-01-01 --end 2023-12-31
"""

import argparse
import os
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import mean_squared_error

TEST_DAYS = 100
LOOKBACK = 60
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def fetch_data(ticker="AAPL", start="2015-01-01", end="2023-12-31"):
    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    close = data["Close"]
    if isinstance(close, pd.DataFrame):  # newer yfinance returns one column per ticker
        close = close.iloc[:, 0]
    close = close.dropna().asfreq("B").ffill()  # Business days
    close.name = "Close"
    return close


def save_plot(name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(os.path.join(OUTPUT_DIR, name), bbox_inches="tight")
    plt.close()


def plot_data(data, ticker):
    plt.figure(figsize=(14, 5))
    plt.plot(data, label="Close Price")
    plt.title(f"{ticker} Closing Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    save_plot("closing_price.png")


def arima_model(data):
    from statsmodels.tsa.arima.model import ARIMA

    train, test = data[:-TEST_DAYS], data[-TEST_DAYS:]
    model_fit = ARIMA(train, order=(5, 1, 0)).fit()
    forecast = model_fit.forecast(steps=TEST_DAYS)

    rmse = sqrt(mean_squared_error(test, forecast))

    plt.figure(figsize=(14, 5))
    plt.plot(test.index, test.values, label="Actual")
    plt.plot(test.index, forecast, label="Forecast", linestyle="--")
    plt.title("ARIMA Forecast")
    plt.legend()
    save_plot("arima_forecast.png")
    return rmse


def prophet_model(data):
    from prophet import Prophet

    df = data.reset_index()
    df.columns = ["ds", "y"]
    train, test = df[:-TEST_DAYS], df[-TEST_DAYS:]

    model = Prophet(daily_seasonality=True)
    model.fit(train)
    future = model.make_future_dataframe(periods=TEST_DAYS)
    forecast = model.predict(future)

    pred = forecast[["ds", "yhat"]].iloc[-TEST_DAYS:]
    rmse = sqrt(mean_squared_error(test["y"].values, pred["yhat"].values))

    model.plot(forecast)
    plt.title("Prophet Forecast")
    save_plot("prophet_forecast.png")
    return rmse


def lstm_model(data, epochs=5):
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow.keras.layers import LSTM, Dense, Input
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    values = data.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    # Fit the scaler and the model on training data only, so the test period stays unseen
    scaler.fit(values[:-TEST_DAYS])
    scaled = scaler.transform(values)

    train = scaled[:-TEST_DAYS]
    X = np.array([train[i - LOOKBACK:i, 0] for i in range(LOOKBACK, len(train))])
    y = np.array([train[i, 0] for i in range(LOOKBACK, len(train))])
    X = X.reshape(X.shape[0], X.shape[1], 1)

    model = Sequential([
        Input(shape=(LOOKBACK, 1)),
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
    model.fit(X, y, epochs=epochs, batch_size=32, verbose=1)

    test_window = scaled[-(TEST_DAYS + LOOKBACK):]
    X_test = np.array([test_window[i - LOOKBACK:i, 0] for i in range(LOOKBACK, len(test_window))])
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    predictions = scaler.inverse_transform(model.predict(X_test, verbose=0))
    actual = values[-TEST_DAYS:]
    rmse = sqrt(mean_squared_error(actual, predictions))

    plt.figure(figsize=(14, 5))
    plt.plot(data.index[-TEST_DAYS:], actual, label="Actual")
    plt.plot(data.index[-TEST_DAYS:], predictions, label="LSTM Forecast", linestyle="--")
    plt.title("LSTM Forecast")
    plt.legend()
    save_plot("lstm_forecast.png")
    return rmse


def main():
    parser = argparse.ArgumentParser(description="Compare ARIMA, Prophet and LSTM stock forecasts")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start", default="2015-01-01")
    parser.add_argument("--end", default="2023-12-31")
    parser.add_argument("--epochs", type=int, default=5, help="LSTM training epochs")
    args = parser.parse_args()

    data = fetch_data(args.ticker, args.start, args.end)
    print(f"Loaded {len(data)} business days of {args.ticker}")
    plot_data(data, args.ticker)

    results = {}
    print("\n--- ARIMA ---")
    results["ARIMA"] = arima_model(data)
    print("\n--- Prophet ---")
    results["Prophet"] = prophet_model(data)
    print("\n--- LSTM ---")
    results["LSTM"] = lstm_model(data, args.epochs)

    print(f"\nRMSE on the last {TEST_DAYS} business days (lower is better):")
    for name, rmse in sorted(results.items(), key=lambda kv: kv[1]):
        print(f"  {name:<8} {rmse:.2f}")
    print(f"\nCharts saved to {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()

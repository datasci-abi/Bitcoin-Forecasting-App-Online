import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

# ===============================
# 1. Load and Prepare the Dataset
# ===============================

# Load the CSV without date parsing
data = pd.read_csv('bitcoin_historical_data.csv')

# Convert the 'Timestamp' column (assumed Unix timestamps in seconds) to datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='s')

# Set the 'Timestamp' as the index and sort the data chronologically
data.set_index('Timestamp', inplace=True)
data.sort_index(inplace=True)

# ===============================
# 2. Exploratory Data Analysis (EDA)
# ===============================

# Print first 5 rows and summary statistics
print("First 5 rows of the dataset:")
print(data.head())
print("\nSummary Statistics:")
print(data.describe())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Plot Bitcoin Closing Price
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Bitcoin Closing Price')
plt.title('Bitcoin Historical Closing Prices')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend(loc='upper right')
plt.show()

# Plot Trading Volume
plt.figure(figsize=(12, 6))
plt.plot(data['Volume'], label='Bitcoin Volume', color='orange')
plt.title('Bitcoin Trading Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend(loc='upper right')
plt.show()

# Plot a 30-Day Moving Average of the Closing Price
data['Close_MA_30'] = data['Close'].rolling(window=30).mean()
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Bitcoin Close')
plt.plot(data['Close_MA_30'], label='30-Day Moving Average', color='red')
plt.title('Bitcoin Closing Price with 30-Day Moving Average')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend(loc='upper right')
plt.show()

# ===============================
# 3. Seasonal Decomposition (using Daily Data)
# ===============================

# Resample the minute-level data to daily frequency (using the last price of each day)
data_daily = data.resample('D').last()

# Check that resampling worked by printing first few rows
print("\nDaily resampled data (first 5 rows):")
print(data_daily.head())

# Perform seasonal decomposition on daily closing prices with a weekly cycle (period=7)
decomposition = seasonal_decompose(data_daily['Close'], model='additive', period=7)
fig = decomposition.plot()
fig.set_size_inches(12, 8)
plt.show()

# ===============================
# 4. Test for Stationarity
# ===============================

result = adfuller(data['Close'].dropna())
print("\nAugmented Dickey-Fuller Test:")
print("ADF Statistic: {:.4f}".format(result[0]))
print("p-value: {:.4f}".format(result[1]))
# A p-value < 0.05 typically indicates that the series is stationary.

# ===============================
# 5. Build an ARIMA Forecasting Model
# ===============================

# Fit an ARIMA model (example order: ARIMA(1, 1, 1))
model = ARIMA(data['Close'], order=(1, 1, 1))
model_fit = model.fit()
print("\nARIMA Model Summary:")
print(model_fit.summary())

# Forecast the next 30 periods (minutes in this case)
forecast = model_fit.forecast(steps=30)
print("\nForecasted Prices for Next 30 Periods:")
print(forecast)

# Create a date index for the forecast (assuming minute-level intervals)
last_date = data.index[-1]
forecast_index = pd.date_range(start=last_date, periods=31, closed='right')

# Plot historical data and forecast
plt.figure(figsize=(12, 6))
plt.plot(data['Close'], label='Historical')
plt.plot(forecast_index, forecast, label='Forecast', color='red')
plt.title('Bitcoin Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend(loc='upper right')
plt.show()

# Save daily resampled data for evaluation
data_daily.to_csv('data_daily.csv')

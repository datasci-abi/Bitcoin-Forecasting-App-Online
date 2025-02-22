import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# -------------------------------
# 1. Load and Preprocess the Data
# -------------------------------
data = pd.read_csv('bitcoin_historical_data.csv')

# Convert 'Timestamp' from Unix timestamp (seconds) to datetime
data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='s')

# Set 'Timestamp' as the index and sort the data
data.set_index('Timestamp', inplace=True)
data.sort_index(inplace=True)

# Resample the data to daily frequency using the last closing price of each day
data_daily = data.resample('D').last()
print("\nDaily resampled data (first 5 rows):")
print(data_daily.head())

# -------------------------------
# 2. Split Data into Train and Test Sets
# -------------------------------
split_point = int(len(data_daily) * 0.8)
train = data_daily.iloc[:split_point]
test = data_daily.iloc[split_point:]
print(f"Training data points: {len(train)}, Test data points: {len(test)}")

# -------------------------------
# 3. Tune ARIMA Model Parameters (p, d, q)
# -------------------------------
best_rmse = float("inf")
best_order = None

# Define a range of values for (p, d, q)
p_values = range(0, 4)  # Try p values from 0 to 3
d_values = range(0, 2)  # Try d values from 0 to 1
q_values = range(0, 4)  # Try q values from 0 to 3

print("\nSearching for the best ARIMA model...")

for p in p_values:
    for d in d_values:
        for q in q_values:
            try:
                order = (p, d, q)
                model = ARIMA(train['Close'], order=order)
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=len(test))
                rmse = np.sqrt(mean_squared_error(test['Close'], forecast))
                print(f"ARIMA{order} RMSE: {rmse}")

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_order = order
            except:
                continue

print(f"\nBest ARIMA Order: {best_order} with RMSE: {best_rmse}")

# -------------------------------
# 4. Train the Best ARIMA Model
# -------------------------------
print(f"\nTraining ARIMA{best_order} model on full dataset...")
best_model = ARIMA(train['Close'], order=best_order)
best_model_fit = best_model.fit()

# -------------------------------
# 5. Forecast Using the Best Model
# -------------------------------
forecast = best_model_fit.forecast(steps=len(test))

# Calculate RMSE
final_rmse = np.sqrt(mean_squared_error(test['Close'], forecast))
print(f"\nFinal Model RMSE: {final_rmse}")

# -------------------------------
# 6. Plot Actual vs Forecasted Prices
# -------------------------------
plt.figure(figsize=(12, 6))
plt.plot(train['Close'], label='Training Data')
plt.plot(test['Close'], label='Test Data', color='green')
plt.plot(test.index, forecast, label='Best Model Forecast', color='red')
plt.title(f'ARIMA{best_order} Forecast vs Actual Prices')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend(loc='upper right')
plt.show()

# -------------------------------
# 7. Forecast the Next 30 Days
# -------------------------------
future_steps = 30  # Number of days to predict
future_forecast = best_model_fit.forecast(steps=future_steps)

# Create future date index
last_date = data_daily.index[-1]
future_dates = pd.date_range(start=last_date, periods=future_steps + 1, freq='D')[1:]

# Plot historical and future forecast
plt.figure(figsize=(12, 6))
plt.plot(data_daily['Close'], label='Historical Prices', color='blue')
plt.plot(future_dates, future_forecast, label='Future Forecast (Next 30 Days)', color='red', linestyle='dashed')
plt.title(f'Bitcoin Price Forecast (Next {future_steps} Days)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend(loc='upper left')
plt.show()

# Print forecasted values
forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted Price': future_forecast})
print("\nNext 30 Days Forecast:")
print(forecast_df)

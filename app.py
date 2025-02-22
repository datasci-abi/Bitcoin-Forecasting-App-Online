import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# -------------------------------
# Load and Preprocess Data
# -------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv('bitcoin_historical_data.csv')
    data['Timestamp'] = pd.to_datetime(data['Timestamp'], unit='s')
    data.set_index('Timestamp', inplace=True)
    data.sort_index(inplace=True)
    data_daily = data.resample('D').last()
    data_daily = data_daily.asfreq('D')  # Fix frequency warning
    return data_daily

# Load data first
data_daily = load_data()

# -------------------------------
# Train ARIMA Model
# -------------------------------
@st.cache_resource
def train_arima(data, order):
    model = ARIMA(data['Close'], order=order)
    return model.fit()

# Define ARIMA order (Replace with best order found earlier)
best_order = (1, 1, 1)

# 🔹 Call train_arima AFTER loading data
model_fit = train_arima(data_daily, best_order)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📈 Bitcoin Price Forecasting App")

# Show historical data
st.subheader("Bitcoin Historical Data")
st.dataframe(data_daily.tail(10))

# Show price trend chart
st.subheader("📊 Bitcoin Price Trend")
st.line_chart(data_daily['Close'])

# -------------------------------
# Forecast Future Prices
# -------------------------------
st.subheader("🔮 Predict Future Bitcoin Prices")
future_days = st.slider("Select number of days to forecast:", 1, 60, 30)

# Generate forecast
future_forecast = model_fit.forecast(steps=future_days)
future_dates = pd.date_range(start=data_daily.index[-1], periods=future_days + 1, freq='D')[1:]

# Display forecast data
forecast_df = pd.DataFrame({'Date': future_dates, 'Forecasted Price': future_forecast})
st.dataframe(forecast_df)

# Plot forecasted trend
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data_daily['Close'], label="Historical Prices", color='blue')
ax.plot(future_dates, future_forecast, label="Forecast (Next Days)", color='red', linestyle='dashed')
ax.set_title(f'Bitcoin Price Forecast for Next {future_days} Days')
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
st.pyplot(fig)

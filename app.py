import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# -------------------------------
# Fetch Real-Time Bitcoin Price
# -------------------------------
def get_live_bitcoin_price():
    url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
    try:
        response = requests.get(url)
        data = response.json()
        return data["bitcoin"]["usd"]
    except Exception as e:
        st.error(f"Error fetching Bitcoin price: {e}")
        return None

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
    data_daily = data_daily.asfreq('D')
    return data_daily

# Load historical data
data_daily = load_data()

# -------------------------------
# Train ARIMA Model
# -------------------------------
@st.cache_resource
def train_arima(data, order):
    model = ARIMA(data['Close'], order=order)
    return model.fit()

best_order = (1, 1, 1)  # Adjust this based on your best model
model_fit = train_arima(data_daily, best_order)

# -------------------------------
# Train XGBoost Model
# -------------------------------
@st.cache_resource
def train_xgboost(data):
    data = data.dropna()  # Remove NaN values
    X = np.arange(len(data)).reshape(-1, 1)  # Time as feature
    y = data['Close'].values

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    # Train the XGBoost model
    model = XGBRegressor(n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    return model, rmse

xgb_model, xgb_rmse = train_xgboost(data_daily)

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("📈 Bitcoin Price Forecasting & Live Tracker")

# 🔹 **Display Live Bitcoin Price**
st.subheader("💰 Live Bitcoin Price (USD)")
live_price = get_live_bitcoin_price()
if live_price:
    st.metric(label="Current BTC Price", value=f"${live_price:,.2f}")

# 📊 **Display Historical Data**
st.subheader("Bitcoin Historical Data")
st.dataframe(data_daily.tail(10))

# 📈 **Show Price Trend**
st.subheader("📊 Bitcoin Price Trend")
st.line_chart(data_daily['Close'])

# -------------------------------
# Forecast Future Prices (ARIMA vs XGBoost)
# -------------------------------
st.subheader("🔮 Predict Future Bitcoin Prices")

future_days = st.slider("Select number of days to forecast:", 1, 60, 30)

# Generate Forecast (ARIMA)
arima_forecast = model_fit.forecast(steps=future_days)
future_dates = pd.date_range(start=data_daily.index[-1], periods=future_days + 1, freq='D')[1:]

# Generate Forecast (XGBoost)
future_X = np.arange(len(data_daily), len(data_daily) + future_days).reshape(-1, 1)
xgb_forecast = xgb_model.predict(future_X)

# Display Forecast
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'ARIMA Forecast': arima_forecast,
    'XGBoost Forecast': xgb_forecast
})
st.dataframe(forecast_df)

# 📉 **Plot Forecast**
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(data_daily['Close'], label="Historical Prices", color='blue')
ax.plot(future_dates, arima_forecast, label="ARIMA Forecast", color='red', linestyle='dashed')
ax.plot(future_dates, xgb_forecast, label="XGBoost Forecast", color='green', linestyle='dotted')
ax.set_title(f'Bitcoin Price Forecast for Next {future_days} Days')
ax.set_xlabel("Date")
ax.set_ylabel("Price (USD)")
ax.legend()
st.pyplot(fig)

# Display RMSE for XGBoost Model
st.subheader("📊 Model Performance")
st.write(f"**XGBoost RMSE:** {xgb_rmse:.2f}")


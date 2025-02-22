# 📈 Bitcoin Forecasting & Live Price Tracker

![Bitcoin Price Prediction](https://your-image-url.com/bitcoin-forecast.png)

## 🚀 Overview
This Streamlit-powered web application provides **real-time Bitcoin price tracking** and **future price predictions** using **ARIMA** and **XGBoost** machine learning models. The app fetches **live Bitcoin prices** from the CoinGecko API and analyzes historical data to generate accurate forecasts.

## ✨ Features
- 🔥 **Live Bitcoin Price Tracking** (from CoinGecko API)
- 📊 **Historical Bitcoin Price Visualization**
- 🔮 **Future Bitcoin Price Prediction (ARIMA & XGBoost)**
- 📈 **Comparison of ARIMA vs. XGBoost Forecasts**
- ✅ **RMSE (Root Mean Squared Error) Calculation for Model Performance**

## 🛠 Installation
Follow these steps to set up the project on your local machine:

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/datasci-abi/Bitcoin-Forecasting-App-Online.git
cd Bitcoin-Forecasting-App-Online
```

### **2️⃣ Create a Virtual Environment & Install Dependencies**
```bash
python -m venv venv
source venv/bin/activate  # For Mac/Linux
venv\Scripts\activate  # For Windows
pip install -r requirements.txt
```

### **3️⃣ Run the Streamlit App**
```bash
streamlit run app.py
```
The app will launch in your browser at `http://localhost:8501`.

## 📡 Live API Integration
This app fetches **real-time Bitcoin prices** from the **CoinGecko API**:
```python
url = "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd"
response = requests.get(url)
data = response.json()
live_price = data["bitcoin"]["usd"]
```

## 📊 Model Details
### **🔹 ARIMA Model** (Time Series Forecasting)
- Used for capturing trends & seasonality in historical Bitcoin prices.
- Model Order: `(1,1,1)` (can be tuned further).

### **🔹 XGBoost Model** (Supervised Learning)
- Uses **past Bitcoin prices** as features to predict future values.
- Trained using **80% historical data**, tested on **20% unseen data**.
- RMSE is calculated to measure accuracy.

## 🚀 Deployment
You can deploy this app on **Streamlit Cloud** by following these steps:
1. Push your code to GitHub.
2. Go to [Streamlit Cloud](https://share.streamlit.io/).
3. Select your GitHub repository & deploy.

## 🏗 Future Improvements
- 📡 **Include more cryptocurrencies (Ethereum, Solana, etc.)**
- 🎯 **Improve forecasting using LSTMs (Deep Learning)**
- 🚀 **Integrate trading signals & notifications**

## 👨‍💻 Author
Developed by **[Abisek RAut](https://github.com/datasci-abi)** ✨

## 📜 License
This project is licensed under the **MIT License**. Feel free to use & modify it!

---
🚀 **Happy Forecasting!** 🚀


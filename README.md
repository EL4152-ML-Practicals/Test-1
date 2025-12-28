# ⚡ Electric Production Time Series Forecasting

> A Machine Learning project for analyzing and forecasting US electric production data using ARIMA model

---

## 🎯 Overview

This project analyzes monthly electric production data in the United States and forecasts future values using time series analysis techniques.

**Key Techniques:**

- 📊 Exploratory Data Analysis (EDA)
- 🔄 Stationarity Testing
- 📈 ARIMA Modeling
- 🔮 Future Forecasting

---

## 📁 Dataset

**File:** `Electric_Production.csv`

- **Time Period:** January 1985 - January 2018
- **Features:**
  - `DATE`: Monthly timestamps
  - `IPG2211A2N`: Electric production index values

---

## 🛠️ Requirements

```python
pandas
matplotlib
statsmodels
```

---

## 💻 Code Walkthrough

### 1️⃣ **Load and Prepare Data** 📂

```python
import pandas as pd
import matplotlib.pyplot as plt

# Read CSV file
df = pd.read_csv('Electric_Production.csv')

# Convert DATE column to datetime
df['DATE'] = pd.to_datetime(df['DATE'])

# Set DATE as index
df.set_index('DATE', inplace=True)
```

**📝 Explanation:** Load the data and convert the DATE column to datetime format for proper time series analysis.

---

### 2️⃣ **Check for Missing Values** 🔍

```python
# Find null values
df.isnull().sum()
```

**📝 Explanation:** Check if there are any missing values in the dataset that need to be handled.

---

### 3️⃣ **Visualize Patterns** 📊

```python
plt.figure(figsize=(10,4))
plt.plot(df, label="Electric Production")
plt.title("Monthly Electric Production in the US")
plt.xlabel("Year")
plt.ylabel("Production")
plt.legend()
plt.show()
```

**📝 Explanation:** Plot the time series to visually identify trends, seasonality, and cyclic patterns.

**🔍 What to look for:**

- 📈 Upward/downward trends
- 🔄 Seasonal patterns (repeating yearly cycles)
- 📉 Sudden drops or spikes

---

### 4️⃣ **Test Stationarity** 🧪

```python
from statsmodels.tsa.stattools import adfuller

# Perform ADF test
result = adfuller(df['IPG2211A2N'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
```

**📝 Explanation:** Use the Augmented Dickey-Fuller (ADF) test to check if data is stationary.

**✅ Stationarity Check:**

- If **p-value < 0.05** → Data is stationary ✔️
- If **p-value > 0.05** → Data is NOT stationary ❌ (needs transformation)

---

### 5️⃣ **Make Data Stationary** 🔧

```python
# Apply differencing
df_diff = df.diff().dropna()

# Test stationarity again
result_diff = adfuller(df_diff['IPG2211A2N'])
print('ADF Statistic (differenced):', result_diff[0])
print('p-value (differenced):', result_diff[1])
```

**📝 Explanation:** Differencing removes trends and makes the data stationary by computing the difference between consecutive observations.

**Formula:** `diff(t) = value(t) - value(t-1)`

---

### 6️⃣ **Build ARIMA Model** 🤖

```python
from statsmodels.tsa.arima.model import ARIMA

# Build ARIMA model with order (p,d,q) = (1,1,1)
model = ARIMA(df, order=(1,1,1))
model_fit = model.fit()

# Forecast next 12 months
forecast = model_fit.forecast(steps=12)
print(forecast)
```

**📝 Explanation:** ARIMA model combines:

- **AR (p=1):** AutoRegressive - uses past values
- **I (d=1):** Integrated - differencing order
- **MA (q=1):** Moving Average - uses past errors

---

### 7️⃣ **Visualize Forecast** 📈

```python
plt.figure(figsize=(10,5))
plt.plot(df, label='Original Data')
plt.plot(forecast, label='Forecasted Data', color='red')
plt.title('Electric Production: Original vs Forecasted')
plt.xlabel('Year')
plt.ylabel('Production')
plt.legend()
plt.show()
```

**📝 Explanation:** Compare actual historical data with predicted future values.

---

## 📊 Results

The ARIMA(1,1,1) model forecasts the next **12 months** of electric production based on historical patterns from 1985-2018.

---

## 🧠 Key Concepts to Remember

| Concept          | Symbol | Meaning                                         |
| ---------------- | ------ | ----------------------------------------------- |
| **Stationarity** | 📏     | Constant mean & variance over time              |
| **Differencing** | ➖     | Removes trend by subtracting consecutive values |
| **ADF Test**     | 🧪     | Tests if data is stationary (p-value < 0.05)    |
| **ARIMA**        | 🤖     | AutoRegressive Integrated Moving Average        |
| **p**            | 🔙     | Number of lag observations (AR order)           |
| **d**            | 🔧     | Degree of differencing                          |
| **q**            | 📊     | Size of moving average window                   |

---

**Made with ❤️ for Machine Learning EL 4152**

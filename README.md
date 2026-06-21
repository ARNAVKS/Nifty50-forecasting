# NIFTY50 Stock Price Forecasting

Time series forecasting of NIFTY50 closing prices using deep learning (LSTM & GRU) on 10 years of historical data (2014–2024).

---

## Results

| Model | MAE | RMSE |
|-------|-----|------|
| **GRU** | **220.80** | **275.42** |
| LSTM | 247.89 | 317.02 |

GRU outperformed LSTM on both metrics. Test set covers May 2023 – Dec 2024.

---

## Pipeline

```
Data Collection (yfinance)
        ↓
EDA — price trend, rolling stats, daily returns
        ↓
Stationarity Tests — ADF & KPSS
        ↓
Feature Engineering — lags, rolling stats, RSI, MACD, Bollinger Bands
        ↓
Train/Val/Test Split (70/15/15, time-ordered)
        ↓
Deep Learning — Stacked LSTM → Stacked GRU
        ↓
Model Comparison + 30-Day Forecast
```

---

## Dataset

- **Source:** Yahoo Finance via `yfinance` (ticker: `^NSEI`)
- **Period:** January 2014 – December 2024
- **Raw rows:** 2,698 trading days
- **After feature engineering:** 2,669 rows × 15 features

**Splits:**
| Split | Period | Rows |
|-------|--------|------|
| Train | Feb 2014 – Sep 2021 | 1,868 |
| Val | Oct 2021 – May 2023 | 400 |
| Test | May 2023 – Dec 2024 | 401 |

---

## Features

| Feature | Description |
|---------|-------------|
| `Close` | Target variable |
| `Volume` | Daily trading volume |
| `lag_1/2/3/5/7/14` | Lagged close prices |
| `rolling_mean_7/30` | 7-day and 30-day rolling mean |
| `rolling_std_7` | 7-day rolling standard deviation |
| `RSI` | Relative Strength Index (14-day) |
| `MACD` | Moving Average Convergence Divergence |
| `BB_high / BB_low` | Bollinger Bands (upper/lower) |

---

## Model Architecture

Both models use a stacked architecture with dropout regularization and are trained on 60-day lookback windows.

**LSTM**
```
LSTM(128, return_sequences=True)  → Dropout(0.2)
LSTM(64)                          → Dropout(0.2)
Dense(32, relu) → Dense(1)
Total params: 118,081
```

**GRU** (same structure, GRU layers instead)
```
GRU(128, return_sequences=True)   → Dropout(0.2)
GRU(64)                           → Dropout(0.2)
Dense(32, relu) → Dense(1)
```

**Training config:**
- Optimizer: Adam (lr=0.001)
- Loss: MSE
- Callbacks: EarlyStopping (patience=10), ReduceLROnPlateau (factor=0.5, patience=5)
- Max epochs: 100
- Batch size: 32

---

## Stationarity Analysis

The raw Close price series is non-stationary (ADF p=0.98, KPSS p=0.01). First differencing makes it stationary (ADF p≈0.00, KPSS p=0.10), confirming I(1) behavior — consistent with financial time series.

---

## 30-Day Forecast

Using the trained GRU model with iterative prediction on a 60-day lookback window:

- **Last known price (Dec 30, 2024):** ₹23,644.90
- **Day-30 forecast:** ₹22,594.61

> Note: Iterative forecasting accumulates error over the horizon. Directional trends are informative; exact values beyond 5–7 days should be interpreted cautiously.

---

## Tech Stack

```
Python 3.12
TensorFlow 2.20 / Keras
yfinance
ta (technical analysis)
statsmodels
scikit-learn
matplotlib / seaborn
pandas / numpy
```

---

## Usage

```bash
pip install yfinance ta statsmodels tensorflow scikit-learn matplotlib seaborn
```

Open `nifty50_clean.ipynb` in Jupyter or Google Colab and run all cells top to bottom.

---

## Project Structure

```
├── nifty50_clean.ipynb   # Main notebook
└── README.md
```

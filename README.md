# Jane Street Market Prediction - LSTM Forecasting Model

This repository contains an LSTM-based time-series forecasting model for the Jane Street market prediction competition. The model is implemented using PyTorch and trained on real-world market data provided in the competition. For more details, visit the competition page: [Jane Street Real-Time Market Data Forecasting](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting)

## Features
- Uses an LSTM neural network to predict target values.
- Incorporates lagged features for improved forecasting.
- Implements a custom R-squared loss function.
- Includes gradient clipping to stabilize training.

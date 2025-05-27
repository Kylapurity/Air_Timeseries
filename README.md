# Air Quality Forecasting Using LSTM

This project focuses on predicting PM2.5 air pollution levels in Beijing using historical weather and air quality data. It was developed for a Kaggle time series competition and explores various deep learning techniques, especially Long Short-Term Memory (LSTM) networks.

## Project Goal

To build and optimize an LSTM model that accurately forecasts PM2.5 concentrations, targeting a scaled RMSE (Root Mean Squared Error) between 0.01 and 0.03.

---

## Dataset

- *Source*: Provided via Kaggle competition.
- *Features*: Includes hourly data for PM2.5, temperature, humidity, wind speed, and more.
- *Time Frame*: Continuous hourly readings ideal for time series forecasting.

---

## Key Steps

### 1. Data Preprocessing
- *Missing Data Handling*: Initially used mean imputation, later switched to interpolation to preserve time patterns.
- *Cyclical Encoding*: Transformed time-related features (like hour and day of week) using sine and cosine functions.
- *Seasonal Decomposition*: Broke down PM2.5 into daily, weekly, and monthly patterns.
- *Sliding Window*: Used a 48-hour input window to predict the next hour’s value.
- *Normalization*: Scaled all features to [0, 1] using MinMaxScaler.

### 2. Model Architecture

- *Base Model*: Recurrent Neural Network using LSTM layers.
- *Final Setup*:
  - 3 Bidirectional LSTM layers with 256, 128, and 64 units.
  - ReLU activation functions.
  - Recurrent Dropout for regularization.
  - Adam optimizer with learning rate tuning.
  - Early Stopping and Batch Normalization in select experiments.

---

## Experiments

A total of *15 experiments* were conducted to optimize model performance. The table below shows key parameters and results for each setup.

### Experiment Summary Table

| Exp | LR     | LSTM Layers         | Units per Layer     | Batch | Dropout          | Seq Len | Activation | Optimizer | MSE     | RMSE   |
|-----|--------|---------------------|----------------------|-------|------------------|---------|------------|-----------|---------|--------|
| 1   | 0.002  | 2                   | 128, 32              | 32    | 0.2              | 48      | ReLU       | Adam      | 0.0114  | 0.0642 |
| 2   | 0.002  | 2                   | 128, 32              | 32    | 0.2              | 24      | ReLU       | Adam      | 0.0132  | 0.0721 |
| 3   | 0.0005 | 2                   | 128, 32              | 32    | 0.2              | 48      | Tanh       | Adam      | 0.0125  | 0.0693 |
| 4   | 0.0005 | 3                   | 256, 128, 64         | 32    | 0.1              | 48      | ReLU       | Adam      | 0.0021  | 0.0458 |
| 5   | 0.0005 | 3                   | 256, 128, 64         | 32    | 0.1              | 24      | ReLU       | Adam      | 0.0035  | 0.0592 |
| 6   | 0.0005 | 3                   | 256, 128, 64         | 32    | 0.1              | 48      | Sigmoid    | Adam      | 0.0006  | 0.0450 |
| 7   | 0.0005 | 2 (Bidirectional)   | 128, 64, 32          | 32    | 0.2              | 48      | ReLU       | Adam      | 0.0018  | 0.0424 |
| 8   | 0.0005 | 2 (Bidirectional)   | 128, 64, 32          | 32    | 0.3              | 48      | Tanh       | Adam      | 0.0023  | 0.0479 |
| 9   | 0.001  | 3                   | 256, 128, 64         | 32    | 0.15             | 48      | Sigmoid    | RMSprop   | 0.0680  | 0.0346 |
| 10  | 0.001  | 2 (Bidirectional)   | 256, 128, 64         | 32    | 0.15             | 24      | ReLU       | Adam      | 0.0070  | 0.0530 |
| 11  | 0.0005 | 3                   | 128, 64, 32          | 32    | Mixed (0.2,0.1)  | 48      | Tanh       | Adam      | 0.0026  | 0.0458 |
| 12  | 0.0005 | 2 (Bidirectional)   | 256, 128, 64         | 32    | Mixed (0.3,0.1)  | 48      | Sigmoid    | Adam      | 0.0007  | 0.0890 |
| 13  | 0.0005 | 3                   | 256, 128, 64         | 32    | 0.1              | 72      | ReLU       | Adam      | 1.2000  | 1.3000 |
| 14  | 0.0005 | 2 (Bidirectional)   | 256, 128, 64         | 32    | Recurrent (0.1)  | 0       | ReLU       | Adam      | 1.4500  | 1.5000 |
| 15  | 0.0005 | 2 (Bidirectional)   | 256, 128, 64         | 32    | Recurrent (0.1)  | 0       | ReLU       | Adam      | 1.1000  | *1.2000* |

---

## Results

- *Best Training RMSE*: 0.0200 (scaled, Experiment 15)
- *Leaderboard RMSE*: 1.2 (higher due to test set pattern shifts)
- *Improvement*: From RMSE 0.0642 (baseline) to 0.0200 after tuning

---

## Challenges Faced

- Handling missing values effectively without breaking time dependencies
- Dealing with cold start prediction issues due to sliding window design
- Generalization gaps between training and leaderboard/test data
- Model tuning required experimentation with architecture and learning rate

---

## Lessons Learned

- Temporal features must be encoded properly for cyclical behavior
- Bidirectional LSTMs and dropout variations improve generalization
- External features like traffic or pollution source data may help further
- Transformers could be explored as a future modeling direction

---

## Getting Started

```bash
git clone https://github.com/Kylapurity/Air_Timeseries.git
cd Air_Timeseries
pip install -r requirements.txt
python scripts/train_model.py

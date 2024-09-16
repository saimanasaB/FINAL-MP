import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings

warnings.filterwarnings('ignore')

# Load the dataset
data_file = "cleaned_data (1).csv"
@st.cache
def load_data(file):
    return pd.read_csv(file)

# Load and preprocess data
data = load_data(data_file)
st.title("Time Series Forecasting: LSTM and SARIMA")

st.write("Dataset Preview:")
st.dataframe(data.head())

# Preprocessing for LSTM
index_column = 'general_index'  # Replace with your actual column name
data = data.set_index('date')
data.index = pd.to_datetime(data.index)
values = data[index_column].values.reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(values)

# LSTM Model
def create_lstm_model(train_data):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(train_data.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_lstm_data(data, time_steps=60):
    X, y = [], []
    for i in range(time_steps, len(data)):
        X.append(data[i - time_steps:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

# Prepare LSTM data
time_steps = 60
X, y = prepare_lstm_data(scaled_data, time_steps)

# Reshape X for LSTM
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Train LSTM model
lstm_model = create_lstm_model(X)
lstm_model.fit(X, y, epochs=10, batch_size=64, verbose=1)

# Predicting for the next 10 years using LSTM
def predict_lstm(model, data, time_steps, n_years=10):
    prediction_input = data[-time_steps:]
    predictions = []
    for _ in range(n_years * 12):  # Assuming monthly data
        pred = model.predict(np.reshape(prediction_input, (1, time_steps, 1)))
        predictions.append(pred[0, 0])
        prediction_input = np.append(prediction_input[1:], pred)
        prediction_input = np.reshape(prediction_input, (time_steps, 1))
    return np.array(predictions)

# Invert scaling for the predictions
lstm_predictions = predict_lstm(lstm_model, scaled_data, time_steps)
lstm_predictions = scaler.inverse_transform(lstm_predictions.reshape(-1, 1))

# Prepare data for SARIMA model
def sarima_forecast(data, order, seasonal_order, steps):
    model = SARIMAX(data, order=order, seasonal_order=seasonal_order)
    result = model.fit()
    forecast = result.forecast(steps=steps)
    return forecast

# Train SARIMA for the next 3 years
sarima_order = (1, 1, 1)  # Adjust based on your dataset
seasonal_order = (1, 1, 1, 12)  # Adjust based on your dataset (assuming monthly seasonality)
sarima_predictions = sarima_forecast(data[index_column], sarima_order, seasonal_order, steps=36)

# Plotting results
st.subheader("LSTM Predictions for Next 10 Years (2025-2035)")
lstm_dates = pd.date_range(start='2025-01-01', periods=len(lstm_predictions), freq='M')
lstm_pred_df = pd.DataFrame(lstm_predictions, index=lstm_dates, columns=['LSTM_Prediction'])

fig, ax = plt.subplots()
ax.plot(data.index, data[index_column], label="Actual")
ax.plot(lstm_pred_df.index, lstm_pred_df['LSTM_Prediction'], label="LSTM Prediction")
plt.legend()
plt.title("LSTM Predictions")
plt.xlabel("Date")
plt.ylabel("General Index")
st.pyplot(fig)

st.subheader("SARIMA Predictions for Next 3 Years (2025-2027)")
sarima_dates = pd.date_range(start='2025-01-01', periods=len(sarima_predictions), freq='M')
sarima_pred_df = pd.DataFrame(sarima_predictions, index=sarima_dates, columns=['SARIMA_Prediction'])

fig, ax = plt.subplots()
ax.plot(data.index, data[index_column], label="Actual")
ax.plot(sarima_pred_df.index, sarima_pred_df['SARIMA_Prediction'], label="SARIMA Prediction")
plt.legend()
plt.title("SARIMA Predictions")
plt.xlabel("Date")
plt.ylabel("General Index")
st.pyplot(fig)


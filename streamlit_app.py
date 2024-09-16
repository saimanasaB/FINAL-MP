import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('cleaned_data.csv')

data = load_data()

# Ensure the 'Generated Date' is a DatetimeIndex
data['Generated Date'] = pd.date_range(start='2020-01-01', periods=len(data), freq='MS')
data.set_index('Generated Date', inplace=True)

st.title("LSTM & SARIMA Forecasting of General Index")

# Display dataset preview
st.write("### Preview of Dataset")
st.write(data.head())

# Display columns in the dataset
st.write("### Columns in the Dataset:")
st.write(data.columns.tolist())

# Check if 'General Index' exists in the dataset
if 'General Index' not in data.columns:
    st.error("Error: 'General Index' column not found in the dataset.")
    st.stop()

# Plot historical data using Altair
st.write("### Historical Data for General Index")
historical_chart = alt.Chart(data.reset_index()).mark_line().encode(
    x='Generated Date:T', y='General Index:Q'
).properties(
    width=700, height=400, title="Historical General Index Data"
)
st.altair_chart(historical_chart)

# Split the data into training, validation, and test sets
train_data = data[:'2022-12']  # Up to December 2022 for training
valid_data = data['2023-01':'2023-12']  # Full year 2023 for validation
test_data = data['2024-01-01':]  # Data from 2024 onwards for testing

# Normalize the General Index for LSTM
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data[['General Index']])
valid_scaled = scaler.transform(valid_data[['General Index']])

# Prepare data for LSTM model
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 12  # Using 12 months (1 year) for sequence length
X_train, y_train = create_sequences(train_scaled, seq_length)
X_valid, y_valid = create_sequences(valid_scaled, seq_length)

# Reshape X_train and X_valid for LSTM model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_valid = np.reshape(X_valid, (X_valid.shape[0], X_valid.shape[1], 1))

# Build the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the LSTM model
history = lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_valid, y_valid))

# Predict future values using LSTM
def predict_lstm(model, last_sequence, future_steps, seq_length):
    predictions = []
    for _ in range(future_steps):
        pred = model.predict(np.reshape(last_sequence, (1, seq_length, 1)))
        predictions.append(pred[0, 0])
        last_sequence = np.append(last_sequence[1:], pred[0, 0]).reshape(seq_length, 1)
    return np.array(predictions).reshape(-1, 1)

future_steps = 10 * 12  # Predicting 10 years (120 months)
last_sequence = train_scaled[-seq_length:]
predictions_lstm = predict_lstm(lstm_model, last_sequence, future_steps, seq_length)

# Inverse transform the LSTM predictions to the original scale
predictions_lstm = scaler.inverse_transform(predictions_lstm)

# Create a DataFrame for LSTM future predictions
future_dates_lstm = pd.date_range(start='2024-03-01', periods=future_steps, freq='MS')
lstm_forecast = pd.DataFrame(predictions_lstm, index=future_dates_lstm, columns=['LSTM Prediction'])

# Evaluate LSTM model
val_predictions_lstm = lstm_model.predict(X_valid)
val_predictions_lstm_rescaled = scaler.inverse_transform(val_predictions_lstm)
lstm_valid_mse = mean_squared_error(valid_data['General Index'][seq_length:], val_predictions_lstm_rescaled)
st.write(f"LSTM Validation MSE: {lstm_valid_mse:.4f}")

# Build the SARIMA model
sarima_model = SARIMAX(train_data['General Index'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_fit = sarima_model.fit(disp=False)

# Predict future values using SARIMA
future_steps_sarima = 3 * 12  # 3 years (36 months)
sarima_forecast = sarima_fit.get_forecast(steps=future_steps_sarima)
sarima_forecast_df = sarima_forecast.conf_int(alpha=0.05)
sarima_forecast_df['SARIMA Prediction'] = sarima_forecast.predicted_mean
sarima_forecast_df.index = pd.date_range(start='2024-03-01', periods=future_steps_sarima, freq='MS')

# Evaluate SARIMA model
sarima_valid_pred = sarima_fit.get_forecast(steps=len(valid_data)).predicted_mean
sarima_valid_mse = mean_squared_error(valid_data['General Index'], sarima_valid_pred)
st.write(f"SARIMA Validation MSE: {sarima_valid_mse:.4f}")

# Plot the LSTM and SARIMA future predictions
st.write("### Future Predictions (LSTM and SARIMA)")
lstm_chart = alt.Chart(lstm_forecast.reset_index()).mark_line(color='blue').encode(
    x='index:T', y='LSTM Prediction:Q'
).properties(width=700, height=400)

sarima_chart = alt.Chart(sarima_forecast_df.reset_index()).mark_line(color='green').encode(
    x='index:T', y='SARIMA Prediction:Q'
).properties(width=700, height=400)

combined_chart = lstm_chart + sarima_chart
st.altair_chart(combined_chart)

# Display prediction data
st.write("### LSTM Predictions for Next 10 Years")
st.write(lstm_forecast)

st.write("### SARIMA Predictions for Next 3 Years")
st.write(sarima_forecast_df[['SARIMA Prediction']])

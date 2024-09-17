import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('cleaned_data.csv')

data = load_data()

st.title("LSTM & SARIMA Forecasting of General index")

# Sidebar for user inputs
st.sidebar.header("Model Configuration")

# LSTM sequence length input
seq_length = st.sidebar.slider("LSTM Sequence Length (months)", min_value=6, max_value=24, value=12, step=1)

# SARIMA order inputs
p = st.sidebar.number_input("SARIMA p (AR order)", min_value=0, max_value=5, value=1, step=1)
d = st.sidebar.number_input("SARIMA d (difference order)", min_value=0, max_value=2, value=1, step=1)
q = st.sidebar.number_input("SARIMA q (MA order)", min_value=0, max_value=5, value=1, step=1)

# SARIMA seasonal order inputs
P = st.sidebar.number_input("SARIMA P (Seasonal AR order)", min_value=0, max_value=5, value=1, step=1)
D = st.sidebar.number_input("SARIMA D (Seasonal difference order)", min_value=0, max_value=2, value=1, step=1)
Q = st.sidebar.number_input("SARIMA Q (Seasonal MA order)", min_value=0, max_value=5, value=1, step=1)
m = st.sidebar.number_input("SARIMA Seasonal period (m)", min_value=1, max_value=12, value=12, step=1)

# Number of years for prediction input
num_years_lstm = st.sidebar.slider("LSTM Prediction Horizon (years)", min_value=1, max_value=10, value=10, step=1)
num_years_sarima = st.sidebar.slider("SARIMA Prediction Horizon (years)", min_value=1, max_value=10, value=3, step=1)

# Display dataset preview
st.write("### Preview of Dataset")
st.write(data.head())

# Display columns in the dataset
st.write("### Columns in the Dataset:")
st.write(data.columns.tolist())

if 'General index' not in data.columns:
    st.error("Error: 'General index' column not found in the dataset.")
    st.stop()

# Generate a time index if no Date column is present
st.write("Since no 'Date' column is provided, generating a time index assuming monthly intervals.")

data['Generated Date'] = pd.date_range(start='2020-01-01', periods=len(data), freq='MS')
data.set_index('Generated Date', inplace=True)

# Plot historical data using Altair
st.write("### Historical Data for General index")
historical_chart = alt.Chart(data.reset_index()).mark_line().encode(
    x='Generated Date:T', y='General index:Q'
).properties(
    width=700, height=400, title="Historical General index Data"
)
st.altair_chart(historical_chart)

# Split data into train and test sets
train_data = data[:'2023']
test_data = data['2024-01-01':]

# Normalize the General index for LSTM
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data[['General index']])

# Prepare data for LSTM model
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Create sequences based on user-configured sequence length
X_train, y_train = create_sequences(train_scaled, seq_length)

# Reshape X_train for LSTM model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Build the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the LSTM model
lstm_model.fit(X_train, y_train, epochs=20, batch_size=32)

# Predict future years using LSTM
future_steps_lstm = num_years_lstm * 12  # Convert years to months
last_sequence = train_scaled[-seq_length:]
predictions_lstm = []

for _ in range(future_steps_lstm):
    pred = lstm_model.predict(np.reshape(last_sequence, (1, seq_length, 1)))
    predictions_lstm.append(pred[0, 0])
    last_sequence = np.append(last_sequence[1:], pred[0, 0]).reshape(seq_length, 1)

# Inverse transform the predictions to original scale
predictions_lstm = scaler.inverse_transform(np.array(predictions_lstm).reshape(-1, 1))

# Create a DataFrame for LSTM future predictions
future_dates_lstm = pd.date_range(start='2024-03-01', periods=future_steps_lstm, freq='MS')
lstm_forecast = pd.DataFrame(predictions_lstm, index=future_dates_lstm, columns=['LSTM Prediction'])

# Build the SARIMA model with user inputs
sarima_model = SARIMAX(train_data['General index'], order=(p, d, q), seasonal_order=(P, D, Q, m))
sarima_fit = sarima_model.fit(disp=False)

# Predict future years using SARIMA
future_steps_sarima = num_years_sarima * 12  # Convert years to months
sarima_forecast = sarima_fit.get_forecast(steps=future_steps_sarima)
sarima_forecast_df = sarima_forecast.conf_int(alpha=0.05)
sarima_forecast_df['SARIMA Prediction'] = sarima_forecast.predicted_mean
sarima_forecast_df.index = pd.date_range(start='2024-03-01', periods=future_steps_sarima, freq='MS')

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
st.write(f"### LSTM Predictions for Next {num_years_lstm} Years")
st.write(lstm_forecast)

st.write(f"### SARIMA Predictions for Next {num_years_sarima} Years")
st.write(sarima_forecast_df[['SARIMA Prediction']])

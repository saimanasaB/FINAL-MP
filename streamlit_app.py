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

st.title("LSTM & SARIMA Forecasting of General Index")

# Display dataset and column names for debugging
st.write("### Preview of Dataset")
st.write(data.head())

# Display column names for clarity
st.write("### Columns in the Dataset")
st.write(data.columns)

# Check if 'Date' column exists, if not adjust accordingly
if 'Date' not in data.columns:
    if 'date' in data.columns:
        st.warning("'Date' column not found, using 'date' instead. Renaming...")
        data.rename(columns={'date': 'Date'}, inplace=True)
    elif 'Timestamp' in data.columns:
        st.warning("'Date' column not found, using 'Timestamp' instead. Renaming...")
        data.rename(columns={'Timestamp': 'Date'}, inplace=True)
    else:
        st.error("Error: 'Date' column not found in dataset. Please check the file.")
        st.write("Columns in the Dataset:", data.columns)
        st.stop()

# Convert the 'Date' column to datetime
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
if data['Date'].isnull().any():
    st.error("Error: Some entries in the 'Date' column could not be converted to datetime. Please check the data.")
    st.write("Rows with invalid Date values:")
    st.write(data[data['Date'].isnull()])
    st.stop()

# Set 'Date' as the index
data.set_index('Date', inplace=True)

# Plot historical data using Altair
st.write("### Historical Data for General Index")
historical_chart = alt.Chart(data.reset_index()).mark_line().encode(
    x=alt.X('Date:T', title='Date'),
    y=alt.Y('General Index:Q', title='General Index'),
    tooltip=['Date:T', 'General Index:Q']
).properties(
    width=700, height=400, title="Historical General Index Data"
)
st.altair_chart(historical_chart)

# Split data into train and test sets
train_data = data[:'2023']
test_data = data['2024-01-01':]

# Normalize the General Index for LSTM
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data[['General Index']])

# Prepare data for LSTM model
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data)-seq_length):
        x = data[i:i+seq_length]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 12  # Using 12 months (1 year) for sequence length
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

# Predict future 10 years using LSTM
future_steps = 10 * 12  # Predicting 10 years (120 months)
last_sequence = train_scaled[-seq_length:]
predictions_lstm = []

for _ in range(future_steps):
    pred = lstm_model.predict(np.reshape(last_sequence, (1, seq_length, 1)))
    predictions_lstm.append(pred[0, 0])
    last_sequence = np.append(last_sequence[1:], pred[0, 0]).reshape(seq_length, 1)

# Inverse transform the predictions to original scale
predictions_lstm = scaler.inverse_transform(np.array(predictions_lstm).reshape(-1, 1))

# Create a DataFrame for LSTM future predictions
future_dates_lstm = pd.date_range(start='2024-03-01', periods=future_steps, freq='MS')
lstm_forecast = pd.DataFrame(predictions_lstm, index=future_dates_lstm, columns=['LSTM Prediction'])

# Build the SARIMA model
sarima_model = SARIMAX(train_data['General Index'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_fit = sarima_model.fit(disp=False)

# Predict future 3 years using SARIMA
future_steps_sarima = 3 * 12  # 3 years (36 months)
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
st.write("### LSTM Predictions for Next 10 Years")
st.write(lstm_forecast)

st.write("### SARIMA Predictions for Next 3 Years")
st.write(sarima_forecast_df[['SARIMA Prediction']])

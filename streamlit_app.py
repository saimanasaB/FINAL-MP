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

# Display dataset preview
st.write("### Preview of Dataset")
st.write(data.head())

# Display columns in the dataset
st.write("### Columns in the Dataset:")
st.write(data.columns.tolist())

# Since 'Date' column is not in the dataset, generate a time index
data['Generated Date'] = pd.date_range(start='2020-01-01', periods=len(data), freq='MS')
data.set_index('Generated Date', inplace=True)

# Plot historical data using Altair
st.write("### Historical Data for General Index")
historical_chart = alt.Chart(data.reset_index()).mark_line().encode(
    x='Generated Date:T', y='General Index:Q'
).properties(
    width=700, height=400, title="Historical General Index Data"
)
st.altair_chart(historical_chart)

# Split data into train and validation sets (80% train, 20% validation)
split_index = int(len(data) * 0.8)
train_data = data.iloc[:split_index]
val_data = data.iloc[split_index:]

# Normalize the General Index for LSTM
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data[['General Index']])
val_scaled = scaler.transform(val_data[['General Index']])

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
X_val, y_val = create_sequences(val_scaled, seq_length)

# Reshape X_train and X_val for LSTM model
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))

# Build the LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the LSTM model with validation
lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

# Predict on validation data
val_predictions = lstm_model.predict(X_val)
val_predictions_rescaled = scaler.inverse_transform(val_predictions)

# Prepare DataFrame for Altair plotting
val_dates = data.index[seq_length + len(train_data):]
val_df = pd.DataFrame({
    'Date': val_dates,
    'Actual': scaler.inverse_transform(y_val.reshape(-1, 1)).flatten(),
    'Predicted': val_predictions_rescaled.flatten()
})

# Plot validation predictions vs actual values using Altair
st.write("### Validation Predictions vs Actual Values")
validation_chart = alt.Chart(val_df).mark_line().encode(
    x='Date:T',
    y='Actual:Q',
    color=alt.value('blue'),
    tooltip=['Date:T', 'Actual:Q']
).properties(
    width=700,
    height=400,
    title="Validation Data and LSTM Predictions"
) + alt.Chart(val_df).mark_line().encode(
    x='Date:T',
    y='Predicted:Q',
    color=alt.value('red'),
    tooltip=['Date:T', 'Predicted:Q']
)
st.altair_chart(validation_chart)

# Build the SARIMA model
sarima_model = SARIMAX(train_data['General Index'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_fit = sarima_model.fit(disp=False)

# Predict future 3 years using SARIMA
future_steps_sarima = 3 * 12  # 3 years (36 months)
sarima_forecast = sarima_fit.get_forecast(steps=future_steps_sarima)
sarima_forecast_df = sarima_forecast.conf_int(alpha=0.05)
sarima_forecast_df['SARIMA Prediction'] = sarima_forecast.predicted_mean
sarima_forecast_df.index = pd.date_range(start='2024-03-01', periods=future_steps_sarima, freq='MS')

# Plot the SARIMA future predictions using Altair
st.write("### SARIMA Future Predictions")
sarima_chart = alt.Chart(sarima_forecast_df.reset_index()).mark_line(color='green').encode(
    x='index:T',
    y='SARIMA Prediction:Q'
).properties(
    width=700, height=400, title="SARIMA Future Predictions"
)
st.altair_chart(sarima_chart)

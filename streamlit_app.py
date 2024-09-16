import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('cleaned_data.csv')

data = load_data()

st.title("LSTM & SARIMA Forecasting of General index with Validation")

# Display dataset preview
st.write("### Preview of Dataset")
st.write(data.head())

# Display columns in the dataset
st.write("### Columns in the Dataset:")
st.write(data.columns.tolist())

# Skip 'Date' column check since it's not in the dataset
# Assume we have a column for 'General index' or similar data
if 'General index' not in data.columns:
    st.error("Error: 'General index' column not found in the dataset.")
    st.stop()

# If there are no dates, generate a simple index for plotting purposes
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

# Split data into train, validation, and test sets
train_data = data[:'2022']
valid_data = data['2023']
test_data = data['2024-01-01':]

# Normalize the General index for LSTM
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data[['General index']])
valid_scaled = scaler.transform(valid_data[['General index']])

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

# Train the LSTM model with validation data
history = lstm_model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_valid, y_valid))

# Plot training and validation loss
st.write("### LSTM Training and Validation Loss")
loss_chart = alt.Chart(pd.DataFrame({
    'epoch': range(1, 21),
    'train_loss': history.history['loss'],
    'val_loss': history.history['val_loss']
})).mark_line().encode(
    x='epoch:Q',
    y=alt.Y('train_loss:Q', title="Train Loss"),
    color=alt.value('blue')
).properties(width=700, height=400)

val_loss_chart = alt.Chart(pd.DataFrame({
    'epoch': range(1, 21),
    'val_loss': history.history['val_loss']
})).mark_line().encode(
    x='epoch:Q',
    y=alt.Y('val_loss:Q', title="Validation Loss"),
    color=alt.value('green')
).properties(width=700, height=400)

st.altair_chart(loss_chart + val_loss_chart)

# Predict future 10 years using LSTM
future_steps = 10 * 12  # Predicting 10 years (120 months)
last_sequence = valid_scaled[-seq_length:]  # Using validation data for prediction
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
sarima_model = SARIMAX(train_data['General index'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_fit = sarima_model.fit(disp=False)

# Predict the validation data using SARIMA
sarima_valid_forecast = sarima_fit.get_prediction(start=valid_data.index[0], end=valid_data.index[-1])
sarima_valid_pred = sarima_valid_forecast.predicted_mean

# Calculate validation MSE for SARIMA
sarima_valid_mse = mean_squared_error(valid_data['General index'], sarima_valid_pred)
st.write(f"SARIMA Validation MSE: {sarima_valid_mse:.4f}")

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

# Validate LSTM model on validation set
st.write("### LSTM Validation on Validation Data")
val_predictions_lstm = lstm_model.predict(X_valid)
val_predictions_lstm_rescaled = scaler.inverse_transform(val_predictions_lstm)

# Calculate LSTM validation MSE
lstm_valid_mse = mean_squared_error(valid_data['General index'][seq_length:], val_predictions_lstm_rescaled)
st.write(f"LSTM Validation MSE: {lstm_valid_mse:.4f}")

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('cleaned_data.csv')

data = load_data()

st.title("LSTM & SARIMA Forecasting of General index")

# Display dataset preview
st.write("### Preview of Dataset")
st.write(data.head())

# Display columns in the dataset
st.write("### Columns in the Dataset:")
st.write(data.columns.tolist())

# Check if 'General index' column exists
if 'General index' not in data.columns:
    st.error("Error: 'General index' column not found in the dataset.")
    st.stop()

# Generate a time index
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
sarima_model = SARIMAX(train_data['General index'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_fit = sarima_model.fit(disp=False)

# Predict future 10 years using SARIMA
future_steps_sarima = 10 * 12  # Predicting 10 years (120 months)
sarima_forecast = sarima_fit.get_forecast(steps=future_steps_sarima)
sarima_forecast_df = sarima_forecast.conf_int(alpha=0.05)
sarima_forecast_df['SARIMA Prediction'] = sarima_forecast.predicted_mean
sarima_forecast_df.index = pd.date_range(start='2024-03-01', periods=future_steps_sarima, freq='MS')

# Metrics Calculation
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100

def weighted_absolute_percentage_error(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100

def median_absolute_percentage_error(y_true, y_pred):
    return np.median(np.abs((y_true - y_pred) / y_true)) * 100

# Evaluate LSTM model
y_true_lstm = test_data['General index'][:future_steps].values
y_pred_lstm = lstm_forecast['LSTM Prediction'].values

mae_lstm = mean_absolute_error(y_true_lstm, y_pred_lstm)
mse_lstm = mean_squared_error(y_true_lstm, y_pred_lstm)
rmse_lstm = np.sqrt(mse_lstm)
mape_lstm = mean_absolute_percentage_error(y_true_lstm, y_pred_lstm)
smape_lstm = symmetric_mean_absolute_percentage_error(y_true_lstm, y_pred_lstm)
wape_lstm = weighted_absolute_percentage_error(y_true_lstm, y_pred_lstm)
mdape_lstm = median_absolute_percentage_error(y_true_lstm, y_pred_lstm)

# Evaluate SARIMA model
y_true_sarima = test_data['General index'][:future_steps_sarima].values
y_pred_sarima = sarima_forecast_df['SARIMA Prediction'].values

mae_sarima = mean_absolute_error(y_true_sarima, y_pred_sarima)
mse_sarima = mean_squared_error(y_true_sarima, y_pred_sarima)
rmse_sarima = np.sqrt(mse_sarima)
mape_sarima = mean_absolute_percentage_error(y_true_sarima, y_pred_sarima)
smape_sarima = symmetric_mean_absolute_percentage_error(y_true_sarima, y_pred_sarima)
wape_sarima = weighted_absolute_percentage_error(y_true_sarima, y_pred_sarima)
mdape_sarima = median_absolute_percentage_error(y_true_sarima, y_pred_sarima)

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

st.write("### SARIMA Predictions for Next 10 Years")
st.write(sarima_forecast_df[['SARIMA Prediction']])

# Metrics Display
st.write("### Model Performance Metrics")

st.write("#### LSTM Model Metrics")
st.write(f"MAE: {mae_lstm:.2f}")
st.write(f"MSE: {mse_lstm:.2f}")
st.write(f"RMSE: {rmse_lstm:.2f}")
st.write(f"MAPE: {mape_lstm:.2f}%")
st.write(f"SMAPE: {smape_lstm:.2f}%")
st.write(f"WAPE: {wape_lstm:.2f}%")
st.write(f"MDAPE: {mdape_lstm:.2f}%")

st.write("#### SARIMA Model Metrics")
st.write(f"MAE: {mae_sarima:.2f}")
st.write(f"MSE: {mse_sarima:.2f}")
st.write(f"RMSE: {rmse_sarima:.2f}")
st.write(f"MAPE: {mape_sarima:.2f}%")
st.write(f"SMAPE: {smape_sarima:.2f}%")
st.write(f"WAPE: {wape_sarima:.2f}%")
st.write(f"MDAPE: {mdape_sarima:.2f}%")

# Custom CSS for better styling
st.markdown("""
    <style>
    .metrics-title {
        font-size: 24px;
        font-weight: bold;
        color: #333;
    }
    .metrics-box {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin-top: 10px;
        background-color: #f9f9f9;
    }
    </style>
    """, unsafe_allow_html=True)

# Display Metrics in Styled Boxes
st.markdown('<div class="metrics-title">LSTM Model Metrics</div>', unsafe_allow_html=True)
st.markdown(f'<div class="metrics-box">MAE: {mae_lstm:.2f}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="metrics-box">MSE: {mse_lstm:.2f}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="metrics-box">RMSE: {rmse_lstm:.2f}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="metrics-box">MAPE: {mape_lstm:.2f}%</div>', unsafe_allow_html=True)
st.markdown(f'<div class="metrics-box">SMAPE: {smape_lstm:.2f}%</div>', unsafe_allow_html=True)
st.markdown(f'<div class="metrics-box">WAPE: {wape_lstm:.2f}%</div>', unsafe_allow_html=True)
st.markdown(f'<div class="metrics-box">MDAPE: {mdape_lstm:.2f}%</div>', unsafe_allow_html=True)

st.markdown('<div class="metrics-title">SARIMA Model Metrics</div>', unsafe_allow_html=True)
st.markdown(f'<div class="metrics-box">MAE: {mae_sarima:.2f}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="metrics-box">MSE: {mse_sarima:.2f}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="metrics-box">RMSE: {rmse_sarima:.2f}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="metrics-box">MAPE: {mape_sarima:.2f}%</div>', unsafe_allow_html=True)
st.markdown(f'<div class="metrics-box">SMAPE: {smape_sarima:.2f}%</div>', unsafe_allow_html=True)
st.markdown(f'<div class="metrics-box">WAPE: {wape_sarima:.2f}%</div>', unsafe_allow_html=True)
st.markdown(f'<div class="metrics-box">MDAPE: {mdape_sarima:.2f}%</div>', unsafe_allow_html=True)


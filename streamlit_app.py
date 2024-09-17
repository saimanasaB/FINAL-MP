import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv('cleaned_data.csv')

data = load_data()

# Apply custom CSS
def add_css():
    st.markdown("""
    <style>
    body {
        font-family: Arial, sans-serif;
    }
    .title {
        text-align: center;
        color: #2E8B57;
        font-size: 36px;
        font-weight: bold;
        margin-top: 20px;
    }
    .section-title {
        color: #4682B4;
        font-size: 28px;
        margin-top: 20px;
        border-bottom: 2px solid #4682B4;
        padding-bottom: 10px;
    }
    .metrics-title {
        font-size: 24px;
        color: #FFA500;
        font-weight: bold;
        text-align: center;
    }
    .metrics-box {
        padding: 10px;
        background-color: #f9f9f9;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    .container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

add_css()

st.markdown('<div class="title">LSTM & SARIMA Forecasting of General Index</div>', unsafe_allow_html=True)

# Display dataset preview
st.write('<div class="section-title">Preview of Dataset</div>', unsafe_allow_html=True)
st.write(data.head())

# Display columns in the dataset
st.write('<div class="section-title">Columns in the Dataset:</div>', unsafe_allow_html=True)
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
st.write('<div class="section-title">Historical Data for General Index</div>', unsafe_allow_html=True)
historical_chart = alt.Chart(data.reset_index()).mark_line().encode(
    x='Generated Date:T', y='General index:Q'
).properties(
    width=700, height=400, title="Historical General Index Data"
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
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the LSTM model
lstm_model.fit(X_train, y_train, epochs=50, batch_size=32)

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

# Predict future 3 years using SARIMA
future_steps_sarima = 3 * 12  # 3 years (36 months)
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
st.write('<div class="section-title">Future Predictions (LSTM and SARIMA)</div>', unsafe_allow_html=True)
lstm_chart = alt.Chart(lstm_forecast.reset_index()).mark_line(color='blue').encode(
    x='index:T', y='LSTM Prediction:Q'
).properties(width=700, height=400, title="LSTM Forecast")

sarima_chart = alt.Chart(sarima_forecast_df.reset_index()).mark_line(color='green').encode(
    x='index:T', y='SARIMA Prediction:Q'
).properties(width=700, height=400,     title="SARIMA Forecast"
)

combined_chart = lstm_chart + sarima_chart
st.altair_chart(combined_chart)

# Display prediction data
st.write('<div class="section-title">LSTM Predictions for Next 10 Years</div>', unsafe_allow_html=True)
st.write(lstm_forecast)

st.write('<div class="section-title">SARIMA Predictions for Next 3 Years</div>', unsafe_allow_html=True)
st.write(sarima_forecast_df[['SARIMA Prediction']])

# Plot evaluation metrics using Altair
st.write('<div class="section-title">Model Evaluation Metrics</div>', unsafe_allow_html=True)

metrics_df = pd.DataFrame({
    "Metric": ["MAE", "MSE", "RMSE", "MAPE", "SMAPE", "WAPE", "MDAPE"],
    "LSTM": [mae_lstm, mse_lstm, rmse_lstm, mape_lstm, smape_lstm, wape_lstm, mdape_lstm],
    "SARIMA": [mae_sarima, mse_sarima, rmse_sarima, mape_sarima, smape_sarima, wape_sarima, mdape_sarima]
})

metrics_melted = metrics_df.melt(id_vars="Metric", var_name="Model", value_name="Value")

metrics_chart = alt.Chart(metrics_melted).mark_bar().encode(
    x=alt.X('Metric:N', title='Metric'),
    y=alt.Y('Value:Q', title='Value'),
    color='Model:N',
    tooltip=['Metric', 'Model', 'Value']
).properties(
    width=700,
    height=400,
    title="Comparison of Model Metrics (LSTM vs SARIMA)"
)

st.altair_chart(metrics_chart)

# Display metrics in tabular format
st.write('<div class="metrics-title">Detailed Metrics Comparison</div>', unsafe_allow_html=True)

st.write("#### LSTM Model Metrics")
st.markdown(f"""
<div class="metrics-box">
<b>MAE:</b> {mae_lstm:.2f} <br/>
<b>MSE:</b> {mse_lstm:.2f} <br/>
<b>RMSE:</b> {rmse_lstm:.2f} <br/>
<b>MAPE:</b> {mape_lstm:.2f}% <br/>
<b>SMAPE:</b> {smape_lstm:.2f}% <br/>
<b>WAPE:</b> {wape_lstm:.2f}% <br/>
<b>MDAPE:</b> {mdape_lstm:.2f}% <br/>
</div>
""", unsafe_allow_html=True)

st.write("#### SARIMA Model Metrics")
st.markdown(f"""
<div class="metrics-box">
<b>MAE:</b> {mae_sarima:.2f} <br/>
<b>MSE:</b> {mse_sarima:.2f} <br/>
<b>RMSE:</b> {rmse_sarima:.2f} <br/>
<b>MAPE:</b> {mape_sarima:.2f}% <br/>
<b>SMAPE:</b> {smape_sarima:.2f}% <br/>
<b>WAPE:</b> {wape_sarima:.2f}% <br/>
<b>MDAPE:</b> {mdape_sarima:.2f}% <br/>
</div>
""", unsafe_allow_html=True)


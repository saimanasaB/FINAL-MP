import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import altair as alt

# Load Data
st.title('LSTM Forecasting of General Index')
st.write('Forecasting from March 2024 to March 2034')

# Load the cleaned CSV file directly
data = pd.read_csv('cleaned_data.csv')

# Check for 'General index' column
if 'General index' not in data.columns:
    st.error("Error: 'General index' column is missing from the data.")
    st.stop()

# Drop the 'sector' column
data = data.drop(columns=['sector'], errors='ignore')

# Display the first few rows of the data
st.subheader('Data Preview')
st.write(data.head())

# Ensure all columns are numeric before applying StandardScaler
numeric_data = data.select_dtypes(include=[np.number])

# Check for NaN values and drop them
if numeric_data.isnull().values.any():
    st.warning("Warning: NaN values found in the data. Dropping these rows.")
    numeric_data = numeric_data.dropna()

# Prepare data for LSTM
st.write("Preparing data for LSTM...")

# Scaling the numeric data using StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

# Split the data into input features (X) and target (y)
X = scaled_data[:, :-1]  # All features except 'General index'
y = scaled_data[:, -1]  # 'General index'

# Reshaping input for LSTM [samples, time steps, features]
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Split the data into training and test sets
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Add hyperparameter inputs
epochs = st.sidebar.number_input('Select number of epochs', min_value=1, max_value=100, value=10)
batch_size = st.sidebar.number_input('Select batch size', min_value=1, max_value=64, value=16)
lstm_units = st.sidebar.number_input('Select LSTM units', min_value=1, max_value=128, value=50)

# Building the LSTM model
model = Sequential()
model.add(LSTM(lstm_units, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))  # Output layer to predict the 'General index'

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model with user-defined hyperparameters
st.write("Training the LSTM model...")
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2)

# Plot training loss
st.subheader('Training Loss')
st.line_chart(history.history['loss'])

# Predicting on the test data
st.write("Evaluating the model on test data...")
predictions = model.predict(X_test)

# Inverse transform the test predictions to the original scale
scaler_general_index = StandardScaler()
scaler_general_index.fit(numeric_data[['General index']])
test_predictions_scaled_back = scaler_general_index.inverse_transform(predictions)
y_test_scaled_back = scaler_general_index.inverse_transform(y_test.reshape(-1, 1))

# Evaluation metrics
mse = mean_squared_error(y_test_scaled_back, test_predictions_scaled_back)
mae = mean_absolute_error(y_test_scaled_back, test_predictions_scaled_back)

st.subheader('Evaluation Metrics')
st.write(f"Mean Squared Error (MSE): {mse:.4f}")
st.write(f"Mean Absolute Error (MAE): {mae:.4f}")

# Plot actual vs predicted values for test data using Altair
test_df = pd.DataFrame({
    'Date': pd.date_range(start='2024-03-01', periods=len(y_test_scaled_back), freq='MS'),
    'Actual': y_test_scaled_back.flatten(),
    'Predicted': test_predictions_scaled_back.flatten()
})

st.subheader('Actual vs Predicted on Test Data')
chart = alt.Chart(test_df).mark_line().encode(
    x=alt.X('Date:T', axis=alt.Axis(tickCount=10)),
    y='Actual:Q',
    color=alt.value('blue')
) + alt.Chart(test_df).mark_line().encode(
    x=alt.X('Date:T', axis=alt.Axis(tickCount=10)),
    y='Predicted:Q',
    color=alt.value('red')
)
st.altair_chart(chart, use_container_width=True)

# Forecasting future values from March 2024 to March 2034
st.write("Forecasting the 'General index'...")

future_steps = 120  # 10 years of monthly data
forecast_input = X[-1].reshape(1, 1, X.shape[2])

forecasted_values = []
for step in range(future_steps):
    forecast = model.predict(forecast_input)
    forecasted_values.append(forecast[0, 0])
    next_input = np.append(forecast_input[0, 0, 1:], forecast)
    forecast_input = next_input.reshape(1, 1, X.shape[2])

# Inverse transform the forecasted values to the original scale of 'General index'
forecasted_values = np.array(forecasted_values).reshape(-1, 1)
forecasted_values_scaled_back = scaler_general_index.inverse_transform(forecasted_values)

# Create future dates
future_dates = pd.date_range(start='2024-03-01', periods=future_steps, freq='MS')

# Create a dataframe with the forecasted results
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Forecasted General index': forecasted_values_scaled_back.flatten()
})

# Display forecasted values
st.subheader('Forecasted General Index')
st.write(forecast_df)

# Plot the forecasted values using Altair
st.subheader('Forecast Plot')
forecast_chart = alt.Chart(forecast_df).mark_line().encode(
    x='Date:T',
    y='Forecasted General index:Q'
)

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import altair as alt

# Load Data
st.title('LSTM Forecasting of General Index')
st.write('Forecasting from March 2024 to March 2034')

# Load the cleaned CSV file directly
data = pd.read_csv('cleaned_data.csv')

# Display the first few rows of the data
st.subheader('Data Preview')
st.write(data.head())

# Assuming 'General index' is the target feature and the rest are predictors
features = data.columns.drop(['General index'])

# Prepare data for LSTM
st.write("Preparing data for LSTM...")

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Split the data into input features (X) and target (y)
X = scaled_data[:, :-1]  # All features except 'General index'
y = scaled_data[:, -1]  # 'General index'

# Reshaping input for LSTM [samples, time steps, features]
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Split the data into training and test sets
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Building the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))  # Output layer to predict the 'General index'

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
st.write("Training the LSTM model...")
model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=2)

# Predicting on the test data
st.write("Evaluating the model on test data...")
predictions = model.predict(X_test)

# Inverse transform to get back to original scale
test_data_scaled_back = scaler.inverse_transform(np.hstack((X_test.reshape(X_test.shape[0], X_test.shape[2]), predictions)))
y_test_scaled_back = scaler.inverse_transform(np.hstack((X_test.reshape(X_test.shape[0], X_test.shape[2]), y_test.reshape(-1, 1))))[:, -1]

# Evaluation metrics
mse = mean_squared_error(y_test_scaled_back, test_data_scaled_back[:, -1])
mae = mean_absolute_error(y_test_scaled_back, test_data_scaled_back[:, -1])

st.subheader('Evaluation Metrics')
st.write(f"Mean Squared Error (MSE): {mse}")
st.write(f"Mean Absolute Error (MAE): {mae}")

# Plot actual vs predicted values for test data using Altair
test_df = pd.DataFrame({
    'Date': pd.date_range(start='2024-03-01', periods=len(y_test_scaled_back), freq='MS'),
    'Actual': y_test_scaled_back,
    'Predicted': test_data_scaled_back[:, -1]
})

st.subheader('Actual vs Predicted on Test Data')
chart = alt.Chart(test_df).mark_line().encode(
    x='Date:T',
    y='Actual:Q',
    color=alt.value('blue')
) + alt.Chart(test_df).mark_line().encode(
    x='Date:T',
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
    # Use the forecasted value as the next input
    next_input = np.append(forecast_input[0, 0, 1:], forecast)
    forecast_input = next_input.reshape(1, 1, X.shape[2])

# Inverse transform to get back to original scale
forecasted_values = np.array(forecasted_values).reshape(-1, 1)
forecasted_values_scaled_back = scaler.inverse_transform(
    np.hstack((scaled_data[:, :-1], forecasted_values))
)[:, -1]

# Create future dates
future_dates = pd.date_range(start='2024-03-01', periods=future_steps, freq='MS')

# Create a dataframe with the forecasted results
forecast_df = pd.DataFrame({
    'Date': future_dates,
    'Forecasted General index': forecasted_values_scaled_back
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
st.altair_chart(forecast_chart, use_container_width=True)

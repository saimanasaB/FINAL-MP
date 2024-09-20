import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import altair as alt
from sklearn.model_selection import KFold
import optuna

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

# Feature Engineering: Add month, year, and lag features
data['Date'] = pd.date_range(start='2024-03-01', periods=len(data), freq='MS')
data['Month'] = data['Date'].dt.month
data['Year'] = data['Date'].dt.year

# Adding lag features for 'General index'
for lag in range(1, 13):  # Creating 12 lag features
    data[f'Lag_{lag}'] = data['General index'].shift(lag)

data = data.drop(columns=['Date'])  # Drop Date if not needed for model
data = data.dropna()  # Drop rows with NaN values after creating lag features

# Display the first few rows of the data
st.subheader('Data Preview')
st.write(data.head())

# Scaling the numeric data using StandardScaler
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Prepare for K-Fold Cross Validation
X = scaled_data[:, :-1]  # All features except 'General index'
y = scaled_data[:, -1]  # 'General index'

# Function to create LSTM model
def create_model(lstm_units):
    model = Sequential()
    model.add(LSTM(lstm_units, return_sequences=False, input_shape=(X.shape[1], 1)))
    model.add(Dense(1))  # Output layer to predict the 'General index'
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Optuna hyperparameter tuning
def objective(trial):
    lstm_units = trial.suggest_int('lstm_units', 1, 128)
    epochs = trial.suggest_int('epochs', 1, 100)
    batch_size = trial.suggest_int('batch_size', 1, 64)
    
    kf = KFold(n_splits=5)
    mse_list = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Reshaping input for LSTM [samples, time steps, features]
        X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

        model = create_model(lstm_units)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

        # Predicting on the test data
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mse_list.append(mse)

    return np.mean(mse_list)

# Run Optuna
study = optuna.create_study(direction='minimize')
st.write("Running hyperparameter tuning...")
study.optimize(objective, n_trials=10)

# Best parameters
best_params = study.best_params
st.subheader('Best Hyperparameters')
st.write(best_params)

# Train final model with best hyperparameters
lstm_units = best_params['lstm_units']
epochs = best_params['epochs']
batch_size = best_params['batch_size']

# Train final model on the full dataset
X = scaled_data[:, :-1].reshape((-1, 1, scaled_data.shape[1] - 1))  # Reshape for LSTM
y = scaled_data[:, -1]

st.write("Training the final LSTM model...")
final_model = create_model(lstm_units)
history = final_model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=2)

# Plot training loss
st.subheader('Training Loss')
st.line_chart(history.history['loss'])

# Predicting on the entire dataset for evaluation
predictions = final_model.predict(X)

# Inverse transform the predictions
scaler_general_index = StandardScaler()
scaler_general_index.fit(data[['General index']])
predictions_scaled_back = scaler_general_index.inverse_transform(predictions)
y_scaled_back = scaler_general_index.inverse_transform(y.reshape(-1, 1))

# Evaluation metrics
mse = mean_squared_error(y_scaled_back, predictions_scaled_back)
mae = mean_absolute_error(y_scaled_back, predictions_scaled_back)

st.subheader('Evaluation Metrics')
st.write(f"Mean Squared Error (MSE): {mse:.4f}")
st.write(f"Mean Absolute Error (MAE): {mae:.4f}")

# Plot actual vs predicted values for the entire dataset using Altair
full_df = pd.DataFrame({
    'Date': pd.date_range(start='2024-03-01', periods=len(y_scaled_back), freq='MS'),
    'Actual': y_scaled_back.flatten(),
    'Predicted': predictions_scaled_back.flatten()
})

st.subheader('Actual vs Predicted on Full Dataset')
chart = alt.Chart(full_df).mark_line().encode(
    x=alt.X('Date:T', axis=alt.Axis(tickCount=10)),
    y='Actual:Q',
    color=alt.value('blue')
) + alt.Chart(full_df).mark_line().encode(
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
    forecast = final_model.predict(forecast_input)
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
st.altair_chart(forecast_chart, use_container_width=True)

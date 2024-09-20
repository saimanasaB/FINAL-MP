import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
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

# Add hyperparameter inputs
epochs = st.sidebar.number_input('Select number of epochs', min_value=1, max_value=100, value=10)
batch_size = st.sidebar.number_input('Select batch size', min_value=1, max_value=64, value=16)
lstm_units = st.sidebar.number_input('Select LSTM units', min_value=1, max_value=128, value=50)
n_splits = st.sidebar.number_input('Select number of folds for cross-validation', min_value=2, max_value=10, value=5)

# Initialize KFold
kf = KFold(n_splits=n_splits)

# Store evaluation metrics for each fold
mse_list = []
mae_list = []

# Cross-validation
st.write("Running cross-validation...")
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Building the LSTM model
    model = Sequential()
    model.add(LSTM(lstm_units, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(Dense(1))  # Output layer to predict the 'General index'

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model with user-defined hyperparameters
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Predicting on the test data
    predictions = model.predict(X_test)

    # Inverse transform the test predictions to the original scale
    scaler_general_index = StandardScaler()
    scaler_general_index.fit(numeric_data[['General index']])
    test_predictions_scaled_back = scaler_general_index.inverse_transform(predictions)
    y_test_scaled_back = scaler_general_index.inverse_transform(y_test.reshape(-1, 1))

    # Calculate evaluation metrics
    mse = mean_squared_error(y_test_scaled_back, test_predictions_scaled_back)
    mae = mean_absolute_error(y_test_scaled_back, test_predictions_scaled_back)

    mse_list.append(mse)
    mae_list.append(mae)

# Average metrics over all folds
avg_mse = np.mean(mse_list)
avg_mae = np.mean(mae_list)

st.subheader('Cross-Validation Results')
st.write(f"Average Mean Squared Error (MSE): {avg_mse:.4f}")
st.write(f"Average Mean Absolute Error (MAE): {avg_mae:.4f}")

# Optionally, you can continue with training on the entire dataset and forecasting
# ...

# You can add the forecasting code here if needed

# Display a message indicating completion
st.write("Cross-validation completed. Adjust the hyperparameters to improve model performance.")

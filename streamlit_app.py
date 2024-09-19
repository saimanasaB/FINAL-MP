import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
import altair as alt
from datetime import timedelta

# Function to load data
@st.cache
def load_data():
    df = pd.read_csv('cleaned_data.csv')
    return df

# Prepare Data for LSTM
def prepare_data(df, feature, look_back):
    # Select the feature to predict (General index)
    data = df[[feature]].values
    
    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    # Create sequences and labels
    X, y = [], []
    for i in range(look_back, len(data_scaled)):
        X.append(data_scaled[i-look_back:i, 0])
        y.append(data_scaled[i, 0])
    
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM

    return X, y, scaler

# Build the LSTM Model
def build_lstm_model(look_back):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Predict future values
def predict_future(model, last_sequence, num_predictions, look_back, scaler):
    future_predictions = []
    current_sequence = last_sequence[-look_back:]
    
    for _ in range(num_predictions):
        prediction = model.predict(current_sequence.reshape(1, look_back, 1), verbose=0)
        future_predictions.append(prediction[0][0])
        current_sequence = np.append(current_sequence[1:], prediction)[-look_back:]
    
    future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
    return future_predictions

# Streamlit App
st.title("General Index Prediction using LSTM")

# Load data
df = load_data()

# Select the feature for prediction
feature = 'General index'

# Display Data
st.write("Data Overview")
st.write(df.head())

# Prepare data for training
look_back = 60  # Look-back period of 60 time steps
X, y, scaler = prepare_data(df, feature, look_back)

# Build and train the model
model = build_lstm_model(look_back)
model.fit(X, y, epochs=10, batch_size=32, verbose=2)

# Predict from March 2024 to March 2034 (10 years)
start_date = pd.to_datetime('2024-03-01')
end_date = pd.to_datetime('2034-03-01')
num_predictions = (end_date - start_date).days // 30  # Approximate months in between

last_sequence = X[-1]
future_predictions = predict_future(model, last_sequence, num_predictions, look_back, scaler)

# Create a dataframe for the future predictions
future_dates = pd.date_range(start=start_date, periods=num_predictions, freq='M')
future_df = pd.DataFrame({'Date': future_dates, 'Predicted General Index': future_predictions.flatten()})

# Display future predictions
st.write("Future Predictions from March 2024 to March 2034")
st.write(future_df)

# Visualize future predictions using Altair
chart = alt.Chart(future_df).mark_line().encode(
    x='Date:T',
    y='Predicted General Index:Q'
).properties(
    title="Predicted General Index (2024-2034)"
)
st.altair_chart(chart, use_container_width=True)

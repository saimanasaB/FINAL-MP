import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import altair as alt

# Load and display the data
st.title("LSTM Model for General Index Prediction")
uploaded_file = "cleaned_data.csv"  # Path to the uploaded file
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", data.head())

    # Select the features to be used for prediction
    features = ['Sector_Rural', 'Sector_Urban', 'Sector_Rural+Urban']
    other_features = [col for col in data.columns if col not in features + ['General index']]
    all_features = features + other_features

    # Preprocess the data
    st.subheader("Preprocessing Data")
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Normalize the features
    data_scaled = scaler.fit_transform(data[all_features + ['General index']])
    st.write(f"Features used for prediction: {all_features}")

    # Prepare the input sequences
    def create_sequences(data, n_steps=12):
        X, y = [], []
        for i in range(len(data) - n_steps):
            X.append(data[i:i+n_steps, :-1])
            y.append(data[i+n_steps, -1])
        return np.array(X), np.array(y)

    n_steps = 12  # Number of past months used to predict the future
    X, y = create_sequences(data_scaled)

    # Train-test split
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    # Build the LSTM model
    st.subheader("Building LSTM Model")
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(n_steps, len(all_features))))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    st.subheader("Training the LSTM Model")
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

    # Make future predictions
    st.subheader("Making Future Predictions")
    def predict_future(model, input_data, n_steps, future_steps):
        predictions = []
        input_seq = input_data[-n_steps:]
        for _ in range(future_steps):
            pred = model.predict(input_seq.reshape(1, n_steps, len(all_features)))
            predictions.append(pred[0][0])
            input_seq = np.append(input_seq[1:], pred, axis=0)
        return np.array(predictions)

    # Define the prediction range (from March 2024 to March 2034, i.e., 10 years, 120 months)
    future_steps = 120
    future_predictions = predict_future(model, data_scaled, n_steps, future_steps)

    # Scale back the predictions to the original range
    scaled_future_predictions = np.zeros((future_steps, data_scaled.shape[1]))
    scaled_future_predictions[:, -1] = future_predictions  # Only setting the General index column
    future_predictions = scaler.inverse_transform(scaled_future_predictions)[:, -1]

    # Prepare future dates for plotting
    dates = pd.date_range(start='2024-03-01', periods=future_steps, freq='MS')

    # Display predictions
    st.subheader("Prediction Results (2024-2034)")
    prediction_df = pd.DataFrame({'Date': dates, 'Predicted General Index': future_predictions})
    st.write(prediction_df)

    # Plot predictions using Altair
    st.subheader("Predicted General Index over Time")
    prediction_chart = alt.Chart(prediction_df).mark_line().encode(
        x='Date:T',
        y='Predicted General Index:Q'
    ).properties(width=700, height=400)

    st.altair_chart(prediction_chart)

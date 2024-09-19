import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import altair as alt
import math

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

    # Hyperparameter Tuning with additional options
    st.sidebar.subheader("Hyperparameter Tuning")
    lstm_units = st.sidebar.slider("Number of LSTM Units", min_value=10, max_value=200, step=10, value=50)
    batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=128, step=16, value=32)
    epochs = st.sidebar.slider("Number of Epochs", min_value=10, max_value=100, step=10, value=20)
    learning_rate = st.sidebar.slider("Learning Rate", min_value=0.0001, max_value=0.01, step=0.0001, value=0.001)
    dropout_rate = st.sidebar.slider("Dropout Rate", min_value=0.0, max_value=0.5, step=0.1, value=0.2)

    # Build the LSTM model with added Dropout and configurable learning rate
    st.subheader("Building LSTM Model")
    model = Sequential()
    model.add(LSTM(lstm_units, return_sequences=True, input_shape=(n_steps, len(all_features))))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(lstm_units))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Train the model
    st.subheader("Training the LSTM Model")
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=2)

    # Model Evaluation on Test Data
    st.subheader("Model Evaluation")
    y_test_pred = model.predict(X_test)

    # Inverse scale the predictions
    y_test_scaled_back = np.zeros((len(y_test), data_scaled.shape[1]))
    y_test_scaled_back[:, -1] = y_test
    y_test_true = scaler.inverse_transform(y_test_scaled_back)[:, -1]

    y_pred_scaled_back = np.zeros((len(y_test_pred), data_scaled.shape[1]))
    y_pred_scaled_back[:, -1] = y_test_pred[:, 0]
    y_pred_true = scaler.inverse_transform(y_pred_scaled_back)[:, -1]

    # Calculate MAE and RMSE
    mae = mean_absolute_error(y_test_true, y_pred_true)
    rmse = math.sqrt(mean_squared_error(y_test_true, y_pred_true))

    st.write(f"Mean Absolute Error (MAE): {mae}")
    st.write(f"Root Mean Squared Error (RMSE): {rmse}")

    # Advanced Error Analysis
    st.subheader("Error Analysis")
    error_df = pd.DataFrame({
        'Date': pd.date_range(start='2020-01-01', periods=len(y_test), freq='MS'),
        'Actual': y_test_true,
        'Predicted': y_pred_true,
        'Error': y_test_true - y_pred_true
    })

    # Plot Actual vs Predicted
    actual_vs_pred_chart = alt.Chart(error_df).mark_line().encode(
        x='Date:T',
        y='Actual:Q',
        color=alt.value("blue")
    ).properties(width=700, height=400).interactive()

    predicted_chart = alt.Chart(error_df).mark_line().encode(
        x='Date:T',
        y='Predicted:Q',
        color=alt.value("red")
    ).properties(width=700, height=400).interactive()

    st.altair_chart(actual_vs_pred_chart + predicted_chart)

    # Plot error over time
    st.subheader("Error Over Time")
    error_chart = alt.Chart(error_df).mark_line().encode(
        x='Date:T',
        y='Error:Q',
        color=alt.value("orange")
    ).properties(width=700, height=400).interactive()

    st.altair_chart(error_chart)

    # Calculate variance of errors for confidence interval estimation
    error_variance = np.var(error_df['Error'])
    confidence_interval = 1.96 * math.sqrt(error_variance)

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

    # Confidence Interval Calculations
    lower_bound = future_predictions - confidence_interval
    upper_bound = future_predictions + confidence_interval

    # Display predictions
    st.subheader("Prediction Results (2024-2034)")
    prediction_df = pd.DataFrame({'Date': dates, 'Predicted General Index': future_predictions, 
                                  'Lower Bound': lower_bound, 'Upper Bound': upper_bound})
    st.write(prediction_df)

    # Plot predictions with confidence intervals using Altair
    st.subheader("Predicted General Index with Confidence Intervals over Time")
    prediction_chart = alt.Chart(prediction_df).mark_line(color='blue').encode(
        x='Date:T',
        y='Predicted General Index:Q'
    )

    lower_bound_line = alt.Chart(prediction_df).mark_line(color='green').encode(
        x='Date:T',
        y='Lower Bound:Q'
    )

    upper_bound_line = alt.Chart(prediction_df).mark_line(color='red').encode(
        x='Date:T',
        y='Upper Bound:Q'
    )

    st.altair_chart(prediction_chart + lower_bound_line + upper_bound_line)

    # Plot Future Predicted Error with a Confidence Interval (Optional Enhancement)
    st.subheader("Prediction Error Confidence Interval")
    future_pred_error_chart = alt.Chart(prediction_df).mark_line().encode(
        x='Date:T',
        y='Predicted General Index:Q',
        color=alt.value("green")
    ).properties(width=700, height=400).interactive()

    st.altair_chart(future_pred_error_chart)

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from keras.models import Sequential
from keras.layers import LSTM, Dense
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
import datetime

# Load the data
st.title("General Index Prediction with LSTM and SARIMA")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())

    # Extracting necessary columns
    if 'date' in df.columns and 'general_index' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        general_index = df['general_index'].values

        # Splitting the data for LSTM and SARIMA
        train_data = df[df.index < '2025']
        train_values = train_data['general_index'].values

        # LSTM Model
        st.subheader("LSTM Model: Predicting 2025 to 2035")
        scaler = MinMaxScaler()
        scaled_train = scaler.fit_transform(train_values.reshape(-1, 1))

        X_train = []
        y_train = []
        time_step = 60  # you can adjust this step
        for i in range(time_step, len(scaled_train)):
            X_train.append(scaled_train[i - time_step:i, 0])
            y_train.append(scaled_train[i, 0])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(LSTM(units=50))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')

        model.fit(X_train, y_train, epochs=10, batch_size=32)

        # Generating future predictions for 2025-2035
        future_predictions = []
        last_sequence = scaled_train[-time_step:]
        for _ in range(10 * 12):  # predicting for 10 years (assuming monthly data)
            last_sequence = last_sequence.reshape((1, time_step, 1))
            next_pred = model.predict(last_sequence)[0, 0]
            future_predictions.append(next_pred)
            last_sequence = np.append(last_sequence[:, 1:, :], [[next_pred]], axis=1)

        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        future_dates = pd.date_range(start='2025-01-01', periods=len(future_predictions), freq='M')
        lstm_df = pd.DataFrame({'date': future_dates, 'predicted_index': future_predictions.flatten()})

        # Plotting LSTM results
        lstm_chart = alt.Chart(lstm_df).mark_line().encode(
            x='date:T',
            y='predicted_index:Q'
        ).properties(
            title='LSTM Predictions (2025-2035)'
        )
        st.altair_chart(lstm_chart)

        # SARIMA Model
        st.subheader("SARIMA Model: Predicting 2025 to 2027")
        sarima_model = SARIMAX(train_values, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
        sarima_results = sarima_model.fit(disp=False)

        future_sarima_predictions = sarima_results.get_forecast(steps=3 * 12).predicted_mean
        sarima_future_dates = pd.date_range(start='2025-01-01', periods=len(future_sarima_predictions), freq='M')
        sarima_df = pd.DataFrame({'date': sarima_future_dates, 'predicted_index': future_sarima_predictions})

        # Plotting SARIMA results
        sarima_chart = alt.Chart(sarima_df).mark_line(color='red').encode(
            x='date:T',
            y='predicted_index:Q'
        ).properties(
            title='SARIMA Predictions (2025-2027)'
        )
        st.altair_chart(sarima_chart)
    else:
        st.error("Make sure the CSV contains 'date' and 'general_index' columns")

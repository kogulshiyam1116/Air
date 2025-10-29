import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Flatten
from tensorflow.keras.optimizers import Adam
import os
import datetime
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# --- Model & Data Configuration ---
DATA_FILE = 'canteen_temperature_data.csv'
SEQ_LENGTH = 24  # Lookback window (as in your .py file)
MIN_DATA_POINTS = 48 # Need at least 24 for one sequence + 24 to train

# --- Helper Functions from your .py file (adapted) ---

def create_sequences(data, seq_length, target_col_idx):
    """Creates sequences for time series forecasting."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, target_col_idx])
    return np.array(X), np.array(y)

def create_attention_lstm_model(seq_length, n_features):
    """Creates the Attention-LSTM model (from your .py file)."""
    inputs = Input(shape=(seq_length, n_features))

    # LSTM layer
    lstm_out = LSTM(64, activation='relu', return_sequences=True)(inputs)
    lstm_out = Dropout(0.2)(lstm_out)

    # Attention mechanism
    attention = Dense(1, activation='tanh')(lstm_out)
    attention = Flatten()(attention)
    attention = tf.keras.activations.softmax(attention)
    attention = tf.keras.layers.RepeatVector(64)(attention)
    attention = tf.keras.layers.Permute([2, 1])(attention)

    # Apply attention
    attended = tf.keras.layers.multiply([lstm_out, attention])
    attended = Flatten()(attended)

    # Output layer
    outputs = Dense(1)(attended)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    return model

# --- Streamlit Data Functions ---

def load_data():
    if os.path.exists(DATA_FILE):
        data = pd.read_csv(DATA_FILE, parse_dates=['Timestamp'])
    else:
        data = pd.DataFrame(columns=['Timestamp', 'Temperature'])
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    return data

def save_data(new_entry):
    new_df = pd.DataFrame([new_entry], columns=['Timestamp', 'Temperature'])
    file_exists = os.path.exists(DATA_FILE)
    new_df.to_csv(DATA_FILE, mode='a', header=not file_exists, index=False)

# --- Streamlit App UI ---

st.set_page_config(layout="wide", page_title="Canteen Temperature Monitor")
st.title("🌡️ Faculty Canteen Temperature Monitor")
st.markdown("This app collects real-time temperature data and predicts the *next hour's* temperature using an Attention-LSTM model.")

# --- 1. Data Collection ---
st.header("1. Log New Temperature Data")

with st.form("temp_form"):
    now = datetime.datetime.now()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        data_date = st.date_input("Date", value=now.date())
    with col2:
        data_time = st.time_input("Time", value=now.time())
    with col3:
        temperature = st.number_input("Temperature (°C)", format="%.2f", step=0.1)

    submitted = st.form_submit_button("Submit Data")

    if submitted:
        try:
            timestamp = datetime.datetime.combine(data_date, data_time)
            new_entry = {'Timestamp': timestamp, 'Temperature': temperature}
            save_data(new_entry)
            st.success(f"Successfully logged: {timestamp} - {temperature}°C")
        except Exception as e:
            st.error(f"An error occurred: {e}")

# --- 2. View Data ---
st.header("2. View Collected Data")

if st.checkbox("Show collected data dashboard"):
    df = load_data()
    if df.empty:
        st.info("No data collected yet. Please submit some data using the form above.")
    else:
        st.subheader("Data History")
        st.dataframe(df.sort_values(by='Timestamp', ascending=False), use_container_width=True)
        
        st.subheader("Temperature Over Time")
        chart_data = df.set_index('Timestamp')['Temperature']
        st.line_chart(chart_data, use_container_width=True)

# --- 3. Prediction (using Attention-LSTM) ---
st.header("3. Prediction (using Attention-LSTM)")
st.info("Note: This model trains on the fly using the collected data. More data will improve accuracy.")

if st.button("Predict Next Hour's Temperature"):
    df = load_data()
    
    if len(df) < MIN_DATA_POINTS:
        st.warning(f"Not enough data to make a prediction. Please collect at least {MIN_DATA_POINTS} data points.")
    else:
        with st.spinner("Training Attention-LSTM model and forecasting..."):
            try:
                # --- 1. Prepare Data ---
                # Resample to hourly and interpolate missing values
                ts_data = df.set_index('Timestamp').resample('H').mean()
                ts_data['Temperature'] = ts_data['Temperature'].interpolate(limit_direction='both')
                ts_data = ts_data.dropna() # Drop any remaining NaNs
                
                if len(ts_data) < MIN_DATA_POINTS:
                    st.warning(f"Not enough sequential hourly data. Need {MIN_DATA_POINTS} hours. Keep collecting data.")
                else:
                    # Scale the data
                    scaler = MinMaxScaler()
                    scaled_data = scaler.fit_transform(ts_data[['Temperature']])
                    n_features = 1 # We are only using Temperature as a feature
                    
                    # --- 2. Create and Train Model ---
                    X_train, y_train = create_sequences(scaled_data, SEQ_LENGTH, 0)
                    
                    model = create_attention_lstm_model(SEQ_LENGTH, n_features)
                    # Train for a few epochs (fast for streamlit)
                    model.fit(X_train, y_train, epochs=10, batch_size=4, verbose=0)
                    
                    # --- 3. Make Prediction ---
                    # Get the last sequence to predict the next step
                    last_sequence_scaled = scaled_data[-SEQ_LENGTH:]
                    last_sequence_scaled = last_sequence_scaled.reshape((1, SEQ_LENGTH, n_features))
                    
                    # Predict
                    pred_scaled = model.predict(last_sequence_scaled)
                    
                    # Inverse transform
                    # We need a dummy array to inverse transform
                    dummy_array = np.zeros((1, n_features))
                    dummy_array[0, 0] = pred_scaled[0][0]
                    prediction = scaler.inverse_transform(dummy_array)[0, 0]

                    last_time = ts_data.index[-1]
                    next_hour = last_time + pd.Timedelta(hours=1)

                    st.success(f"**Predicted Temperature for {next_hour.strftime('%Y-%m-%d %I:%M %p')}:**")
                    st.metric(label="Predicted Temperature", value=f"{prediction:.2f} °C")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.exception(e)


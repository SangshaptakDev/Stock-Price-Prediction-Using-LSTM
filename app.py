import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow.keras.models import load_model

# Define the directory where stock data is stored
DATA_DIRECTORY = "C:/STOCK PRICE PREDICTION MODEL COMPANY"
MODEL_PATH = os.path.join(DATA_DIRECTORY, "lstm_model.h5")

# Load available stock files
stock_files = [f for f in os.listdir(DATA_DIRECTORY) if f.endswith(".csv")]
stock_tickers = [f.replace(".csv", "") for f in stock_files]

# ✅ Custom CSS to control background dynamically
default_background = "https://4kwallpapers.com/images/walls/thumbs_2t/13849.png"
graph_background = "https://wallpaperaccess.com/full/806427.jpg"

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{default_background}");
        background-size: cover;
        background-attachment: fixed;
        transition: background-image 1s ease-in-out;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.title("📈 Stock Price Prediction using LSTM")
st.sidebar.header("Select Company & Date Range")

# Select company
selected_company = st.sidebar.selectbox("Choose a company:", stock_tickers)

# Select Date Range (Default set from 2019-11-30 to 2024-10-30)
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2019-11-30"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-10-30"))

# Add "Enter" button to trigger prediction
if st.sidebar.button("🔍 Enter to Predict"):
    # Load the selected company's data
    file_path = os.path.join(DATA_DIRECTORY, f"{selected_company}.csv")
    df = pd.read_csv(file_path, parse_dates=["Date"])

    # Filter data within the selected date range
    df = df[(df["Date"] >= str(start_date)) & (df["Date"] <= str(end_date))]

    # Check if data is available
    if df.empty:
        st.error("❌ No data available for the selected date range. Try a different range.")
        st.stop()

    # Define the 5 features (to match trained model)
    features = ["Open", "High", "Low", "Close", "Volume"]

    # Scale the features
    scaler_X = StandardScaler()
    scaler_Y = MinMaxScaler()

    X = scaler_X.fit_transform(df[features])
    y = scaler_Y.fit_transform(df[["Close"]])

    # Reshape for LSTM (samples, time-steps, features)
    X = X.reshape(X.shape[0], 1, X.shape[1])

    # Load the trained model
    model = load_model(MODEL_PATH, compile=False)
    model.compile(optimizer="adam", loss="mean_squared_error")

    # Ensure input shape matches the model's expected shape
    if model.input_shape[2] != X.shape[2]:
        st.error(f"❌ Feature mismatch! Model expects {model.input_shape[2]} features, but got {X.shape[2]}.")
        st.stop()

    # Predict stock prices
    y_pred_scaled = model.predict(X)
    y_pred = scaler_Y.inverse_transform(y_pred_scaled)
    y_actual = scaler_Y.inverse_transform(y)

    # ✅ Change Background to New Image After Prediction
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("{graph_background}") !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    # Transparent Graph with Colored Axis Labels
    fig, ax = plt.subplots(figsize=(12, 6), facecolor="none")  # Transparent figure
    ax.patch.set_alpha(1)  # Adjust graph transparency

    # Plot Actual vs Predicted Prices
    ax.plot(df["Date"], y_actual, label="Actual Prices", color="white", linestyle="-", linewidth=2)
    ax.plot(df["Date"], y_pred, label="Predicted Prices", color="red", linestyle="dashed", linewidth=2)

    # Customize Axis Labels, Ticks, and Title
    ax.set_xlabel("Date", fontsize=12, color="white")  # X-axis label color
    ax.set_ylabel("Stock Price", fontsize=12, color="white")  # Y-axis label color
    ax.set_title(f"Actual vs Predicted Stock Prices - {selected_company}", color="cyan", fontsize=14)  # Title color

    ax.tick_params(axis="x", colors="white")  # X-axis ticks color
    ax.tick_params(axis="y", colors="white")  # Y-axis ticks color

    ax.legend()
    ax.grid()

    # Display transparent graph in Streamlit
    st.pyplot(fig, transparent=True)

    st.success("✅ Prediction Completed Successfully!")

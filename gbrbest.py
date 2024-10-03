import streamlit as st
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
import joblib

# Set a random seed for consistency
np.random.seed(42)
random.seed(42)

# Load your dataset
df = pd.read_csv('C://Users//Nitin//model_deployment_solarpanel//solarpowergeneration.csv')
df.fillna(df.mean(), inplace=True)  # Handle missing values

# Defining the target and features
Y = df['power-generated']  # Ensure the column name matches exactly
X = df.drop(columns=['power-generated'])

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

# Standardizing the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train your Gradient Boosting model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=4, min_samples_split=5)
model.fit(X_train_scaled, y_train)

# Save the model for later use
joblib.dump(model, 'solar_power_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Evaluate the model
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)
training_acc = r2_score(y_train, y_pred_train)
test_acc = r2_score(y_test, y_pred_test)

# Display the local image in the background
image_path = "C://Users//Nitin//Downloads//solarimage.jpeg"
st.image(image_path, use_column_width=True)

# Overlay content using st.markdown
st.markdown("<h1 style='text-align: center; color: white;'>Solar Power Prediction</h1>", unsafe_allow_html=True)

# Display model performance
st.sidebar.header("Model Performance")
st.sidebar.write(f"Training R²: {training_acc:.2f}")
st.sidebar.write(f"Test R²: {test_acc:.2f}")

# Input fields for features in a sidebar
st.sidebar.header('User Input Parameters')
distance_to_solar_noon = st.sidebar.number_input("Distance to Solar Noon")
temperature = st.sidebar.number_input("Temperature")
wind_direction = st.sidebar.number_input("Wind Direction")
wind_speed = st.sidebar.number_input("Wind Speed")
sky_cover = st.sidebar.number_input("Sky Cover")
visibility = st.sidebar.number_input("Visibility")
humidity = st.sidebar.number_input("Humidity")
average_wind_speed = st.sidebar.number_input("Average Wind Speed (Period)")
average_pressure = st.sidebar.number_input("Average Pressure (Period)")

# Create a DataFrame from user input
input_data = pd.DataFrame({
    "distance-to-solar-noon": [distance_to_solar_noon],
    "temperature": [temperature],
    "wind-direction": [wind_direction],
    "wind-speed": [wind_speed],
    "sky-cover": [sky_cover],
    "visibility": [visibility],
    "humidity": [humidity],
    "average-wind-speed-(period)": [average_wind_speed],
    "average-pressure-(period)": [average_pressure]
})

# Ensure the input_data columns are in the same order as X_train
input_data = input_data[X_train.columns]

# Standardize the input data
input_data_scaled = scaler.transform(input_data)

# Make prediction when button is clicked
if st.button("Predict"):
    prediction = model.predict(input_data_scaled)
    prediction = max(0, prediction[0])  # Ensure the prediction is non-negative
    st.success(f"Predicted Power Generation: {prediction:.2f} MW")


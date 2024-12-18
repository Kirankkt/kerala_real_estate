import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model and feature names
model = joblib.load('model.pkl')
location_encoder = joblib.load('location_encoder.pkl')

# Streamlit app
st.title("Real Estate Price Prediction")

st.sidebar.header("Input Parameters")
build_area = st.sidebar.number_input("Built-up Area (sq ft)", min_value=500.0, max_value=10000.0, value=1500.0)
plot_area_cents = st.sidebar.number_input("Plot Area (cents)", min_value=1.0, max_value=50.0, value=4.0)
bedrooms = st.sidebar.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
location = st.sidebar.selectbox("Location", options=location_encoder.categories_[0])

# Function for predictions
def predict_price(build_area, plot_area_cents, bedrooms, location):
    build_to_plot_ratio = build_area / (plot_area_cents * 435.6)
    total_area = build_area + (plot_area_cents * 435.6)
    
    # Create input array with one-hot encoded location
    location_vector = [0] * len(location_encoder.get_feature_names_out())
    location_features = location_encoder.transform([[location]]).toarray()[0]
    input_data = np.array([[build_area, plot_area_cents, bedrooms, build_to_plot_ratio, total_area] + location_features.tolist()])
    
    predicted_price = model.predict(input_data)[0]
    return predicted_price

if st.sidebar.button("Predict Price"):
    price = predict_price(build_area, plot_area_cents, bedrooms, location)
    st.write(f"The predicted price is ₹{price:,.2f}")

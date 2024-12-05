import streamlit as st
import joblib
import os
import pandas as pd

# Get the directory of the current file
base_dir = os.path.dirname(os.path.abspath(__file__))
print(base_dir)
# Construct the relative path
relative_path = os.path.join(base_dir, 'model', 'fare_prediction_model.pkl')
# Load your trained model
model = joblib.load(relative_path)
# Load the DataFrame
csv_path = os.path.join(os.path.dirname(__file__), '../../Datasets/preprocessed.csv')

# Read the CSV file
df = pd.read_csv(csv_path)

# Extract unique values for categorical features
car_conditions = df['Car Condition'].unique()
weathers = df['Weather'].unique()
traffic_conditions = df['Traffic Condition'].unique()

# Streamlit app
st.title('Taxi Fare Prediction')

# Input features
car_condition = st.selectbox('Car Condition', car_conditions)
weather = st.selectbox('Weather', weathers)
traffic_condition = st.selectbox('Traffic Condition', traffic_conditions)
pickup_datetime = st.text_input('Pickup Datetime (YYYY-MM-DD HH:MM:SS)')
pickup_longitude = st.number_input('Pickup Longitude', format="%.6f")
pickup_latitude = st.number_input('Pickup Latitude', format="%.6f")
dropoff_longitude = st.number_input('Dropoff Longitude', format="%.6f")
dropoff_latitude = st.number_input('Dropoff Latitude', format="%.6f")
passenger_count = st.number_input('Passenger Count', min_value=1, max_value=10, step=1)
hour = st.number_input('Hour', min_value=0, max_value=23, step=1)
day = st.number_input('Day', min_value=1, max_value=31, step=1)
month = st.number_input('Month', min_value=1, max_value=12, step=1)
weekday = st.number_input('Weekday', min_value=0, max_value=6, step=1)
year = st.number_input('Year', min_value=2000, max_value=2100, step=1)
jfk_dist = st.number_input('Distance to JFK', format="%.2f")
ewr_dist = st.number_input('Distance to EWR', format="%.2f")
lga_dist = st.number_input('Distance to LGA', format="%.2f")
sol_dist = st.number_input('Distance to Statue of Liberty', format="%.2f")
nyc_dist = st.number_input('Distance to NYC', format="%.2f")
distance = st.number_input('Trip Distance', format="%.2f")
bearing = st.number_input('Bearing', format="%.2f")

# Create a DataFrame for the input features
input_data = pd.DataFrame({
    'Car Condition': [car_condition],
    'Weather': [weather],
    'Traffic Condition': [traffic_condition],
    'pickup_datetime': [pickup_datetime],
    'pickup_longitude': [pickup_longitude],
    'pickup_latitude': [pickup_latitude],
    'dropoff_longitude': [dropoff_longitude],
    'dropoff_latitude': [dropoff_latitude],
    'passenger_count': [passenger_count],
    'hour': [hour],
    'day': [day],
    'month': [month],
    'weekday': [weekday],
    'year': [year],
    'jfk_dist': [jfk_dist],
    'ewr_dist': [ewr_dist],
    'lga_dist': [lga_dist],
    'sol_dist': [sol_dist],
    'nyc_dist': [nyc_dist],
    'distance': [distance],
    'bearing': [bearing]
})

# Predict the fare
if st.button('Predict Fare'):
    prediction = model.predict(input_data)
    st.write(f'Predicted Fare: ${prediction[0]:.2f}')
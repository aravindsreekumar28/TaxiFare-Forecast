import streamlit as st
import joblib
import os
import pandas as pd

# CSS for styling
def add_css():
    st.markdown(
        """
        <style>
        .main {background-color: #f0f2f6;}
        .stButton>button {background-color: #4CAF50; color: white; border-radius: 8px; padding: 10px 20px; border: none; margin-top: 20px; cursor: pointer;}
        .stTextInput, .stNumberInput, .stSelectbox {margin-bottom: 10px; border-radius: 8px; border: 1px solid #ddd; padding: 10px;}
        .stTitle {font-size: 28px; font-weight: bold; color: #333; text-align: center; margin-bottom: 40px;}
        .container {display: flex; flex-direction: column; align-items: center; justify-content: center; background-color: #ffffff; padding: 20px; border-radius: 10px; box-shadow: 0px 0px 15px rgba(0,0,0,0.2); max-width: 800px; margin: auto;}
        .input-block {width: 100%; max-width: 600px; margin: 10px 0;}
        .header {color: #4CAF50;}
        </style>
        """,
        unsafe_allow_html=True
    )

add_css()

# Get the directory of the current file
base_dir = os.path.dirname(os.path.abspath(__file__))
# Construct the relative path
relative_path = os.path.join(base_dir, 'Notebooks', 'Data Modelling','model', 'fare_prediction_model.pkl')
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

# Define all possible categorical feature names
possible_features = [
    'pickup_longitude', 'pickup_latitude', 'dropoff_longitude',
    'dropoff_latitude', 'passenger_count', 'hour', 'day', 'month',
    'weekday', 'year', 'jfk_dist', 'ewr_dist', 'lga_dist',
    'sol_dist', 'nyc_dist', 'distance', 'bearing',
    'Car Condition_Excellent', 'Car Condition_Good', 'Car Condition_Very Good',
    'Weather_rainy', 'Weather_stormy', 'Weather_sunny', 'Weather_windy',
    'Traffic Condition_Dense Traffic', 'Traffic Condition_Flow Traffic'
]

# Streamlit app
st.markdown('<div class="stTitle">Taxi Fare Prediction</div>', unsafe_allow_html=True)

with st.form(key='fare_form'):
    st.subheader('Enter Trip Details', anchor=None)
    
    # Input features
    car_condition = st.selectbox('Car Condition', car_conditions, key='car_condition', help='Select the condition of the car.')
    weather = st.selectbox('Weather', weathers, key='weather', help='Select the weather during the trip.')
    traffic_condition = st.selectbox('Traffic Condition', traffic_conditions, key='traffic_condition', help='Select the traffic condition during the trip.')
    pickup_longitude = st.number_input('Pickup Longitude', format="%.6f", help='Enter the longitude for pickup location.')
    pickup_latitude = st.number_input('Pickup Latitude', format="%.6f", help='Enter the latitude for pickup location.')
    dropoff_longitude = st.number_input('Dropoff Longitude', format="%.6f", help='Enter the longitude for dropoff location.')
    dropoff_latitude = st.number_input('Dropoff Latitude', format="%.6f", help='Enter the latitude for dropoff location.')
    passenger_count = st.number_input('Passenger Count', min_value=1, max_value=10, step=1, help='Enter the number of passengers.')
    hour = st.number_input('Hour', min_value=0, max_value=23, step=1, help='Enter the hour of the day for the trip.')
    day = st.number_input('Day', min_value=1, max_value=31, step=1, help='Enter the day of the month for the trip.')
    month = st.number_input('Month', min_value=1, max_value=12, step=1, help='Enter the month of the year for the trip.')
    weekday = st.number_input('Weekday', min_value=0, max_value=6, step=1, help='Enter the day of the week for the trip.')
    year = st.number_input('Year', min_value=2000, max_value=2100, step=1, help='Enter the year of the trip.')
    jfk_dist = st.number_input('Distance to JFK', format="%.2f", help='Enter the distance to JFK airport.')
    ewr_dist = st.number_input('Distance to EWR', format="%.2f", help='Enter the distance to EWR airport.')
    lga_dist = st.number_input('Distance to LGA', format="%.2f", help='Enter the distance to LGA airport.')
    sol_dist = st.number_input('Distance to Statue of Liberty', format="%.2f", help='Enter the distance to the Statue of Liberty.')
    nyc_dist = st.number_input('Distance to NYC', format="%.2f", help='Enter the distance to NYC.')
    distance = st.number_input('Trip Distance', format="%.2f", help='Enter the total trip distance.')
    bearing = st.number_input('Bearing', format="%.2f", help='Enter the bearing angle for the trip.')

    # Submit button
    submit_button = st.form_submit_button(label='Predict Fare')

# Create a DataFrame for the input features
if submit_button:
    input_data = pd.DataFrame({
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

    # One-hot encode categorical features
    categorical_data = {
        f'Car Condition_{car_condition}': [1],
        f'Weather_{weather}': [1],
        f'Traffic Condition_{traffic_condition}': [1]
    }

    # Ensure all one-hot encoded features are included in the input_data
    for feature in ['Car Condition_Excellent', 'Car Condition_Good', 'Car Condition_Very Good', 'Weather_rainy', 'Weather_stormy', 'Weather_sunny', 'Weather_windy', 'Traffic Condition_Dense Traffic', 'Traffic Condition_Flow Traffic']:
        if feature not in categorical_data:
            categorical_data[feature] = [0]

    # Combine the input features and ensure all necessary features are present
    input_data = input_data.join(pd.DataFrame(categorical_data))

    # Add missing features with zero values
    for feature in possible_features:
        if feature not in input_data:
            input_data[feature] = 0

    # Reorder columns to match model's expected input
    input_data = input_data[possible_features]

    # Predict the fare
    prediction = model.predict(input_data)
    st.write(f'Predicted Fare: <b style="color: #4CAF50;">${prediction[0]:.2f}</b>', unsafe_allow_html=True)

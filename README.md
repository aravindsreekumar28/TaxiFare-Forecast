# Taxi Fare Prediction Project

This project is a collaborative effort to predict taxi fares using machine learning techniques. By analyzing historical taxi fare data, we aim to build a model that can predict the fare based on various factors such as distance, time, and location details.

## Project Overview

The project uses a machine learning model to forecast taxi fares, leveraging historical data to improve fare accuracy predictions. This is a joint endeavor by developers **Rajib, Aravind, Srividhya**, and **Kavipriya**.

- **GitHub Link to the Notebook:** [Shared Data Science Project - Taxi Fare Prediction](https://github.com/aravindsreekumar28/TaxiFare-Forecast/blob/main/Notebooks/Shared_DS_Project_Taxi_Fare_Prediction.ipynb)
- **Dataset Link:** [Taxi Fare Dataset - final.csv](https://github.com/aravindsreekumar28/TaxiFare-Forecast/blob/main/Datasets/final.csv)

## Dataset

The dataset used for this project includes comprehensive details necessary for building a robust fare prediction model. It is located at the above link and should be downloaded and preprocessed as required before model training.

### Key Features in the Dataset
- **Pickup and Dropoff Times:** These help to account for traffic patterns during different times of the day.
- **Geographical Coordinates:** Pickup and dropoff locations are crucial for calculating distances and estimating fares.
- **Distance:** The distance traveled, calculated from pickup and dropoff coordinates, is a primary predictor of the fare.
- **Additional Factors:** Other fields may include the number of passengers, vendor ID, and more to enhance the model's accuracy.

## Notebooks

The primary notebook, available [here](https://github.com/aravindsreekumar28/TaxiFare-Forecast/blob/main/Notebooks/Shared_DS_Project_Taxi_Fare_Prediction.ipynb), details the data exploration, preprocessing, model training, and evaluation steps. Please follow the order in the notebook to understand each step of the model-building process.

## Installation and Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/aravindsreekumar28/TaxiFare-Forecast.git
   ```
   
2. **Navigate to the Notebook Directory:**
   ```bash
   cd TaxiFare-Forecast/Notebooks
   ```

3. **Install Required Libraries:**
   Ensure all required packages are installed by running the following command:
   ```bash
   pip install -r requirements.txt
   ```

4. **Load the Dataset:**
   Make sure the dataset is placed in the appropriate directory as referenced in the notebook for smooth execution.

## Model Training

The notebook goes through the steps of:
1. Data Cleaning and Preprocessing
2. Feature Engineering
3. Model Selection and Training
4. Model Evaluation

Each step is explained within the notebook, providing insights into the choices made for improving prediction accuracy.

## Contributors

- **Rajib**
- **Aravind**
- **Srividhya**
- **Kavipriya**

Feel free to reach out to the contributors for any questions or further insights regarding the project.


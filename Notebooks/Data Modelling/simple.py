import streamlit as st
import sys
import os

st.title('Streamlit Diagnostic App')

# System Information
st.header('System Information')
st.write(f"Python Version: {sys.version}")
st.write(f"Current Working Directory: {os.getcwd()}")
st.write(f"Script Location: {os.path.abspath(__file__)}")

# Environment Check
st.header('Environment Check')
try:
    import pandas
    import joblib
    import sklearn

    st.success('Required libraries are installed:')
    st.write(f"Pandas version: {pandas.__version__}")
    st.write(f"Joblib version: {joblib.__version__}")
    st.write(f"Scikit-learn version: {sklearn.__version__}")
except ImportError as e:
    st.error(f"Library import error: {e}")

# Interactive Section
st.header('Interactive Test')
name = st.text_input('Enter your name:')
if name:
    st.write(f'Hello, {name}!')

age = st.slider('Select your age', 0, 100, 25)
st.write(f'You selected: {age} years old')

# Simple Calculation
st.header('Simple Calculation')
num1 = st.number_input('Enter first number', value=0)
num2 = st.number_input('Enter second number', value=0)

if st.button('Add Numbers'):
    result = num1 + num2
    st.write(f'Result: {result}')
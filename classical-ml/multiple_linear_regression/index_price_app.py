import streamlit as st
import requests

st.title('Index Price Predictor')

interest_rate = st.number_input('Enter Interest Rate')
unemployment_rate = st.number_input('Enter Unemployment Rate')

if st.button('Predict'):
    response = requests.post(
        "http://localhost:8000/predict",
        json={
            "interest_rate": interest_rate,
            "unemployment_rate": unemployment_rate
        }
    )
    result = response.json()['index_price']
    st.success(f"Predicted Price: {result}")
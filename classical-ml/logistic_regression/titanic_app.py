import streamlit as st
import requests

st.title("Titanic Survival Prediction")

st.write("Predict whether a passenger would survive")

pclass = st.selectbox("Passenger Class",[1,2,3])
sex = st.selectbox("Sex",['Female','Male'])
age = st.slider("Age",1,80,25)
sibsp = st.number_input("Siblings/Spouses",0,10,0)
parch = st.number_input("Parents/Children",0,10,0)
fare = st.number_input("Fare",0,600,50)
embarked = st.selectbox("Embarked",["C","Q","S"])

if st.button("Predict Survival"):
    sex_value = 0 if sex == "Female" else 1
    embarked_mapping = {"C":0, "Q":1, "S":2}
    
    payload = {
        "pclass" : pclass,
        "sex" : sex_value,
        "age" : age,
        "sibsp" : sibsp,
        "parch" : parch,
        "fare" : fare,
        "embarked" : embarked_mapping[embarked]
    }
    
    response = requests.post("http://127.0.0.1:8000/predict", json=payload)
    result = response.json()
    
    if result['survived'] == 1:
        st.success(f"Passenger Survived\n\nSurvial Probability:{result['survival_probability']}")
    else:
        st.error(f"Passenger Did Not Survive\n\nSurvival Probability:{result['survival_probability']}")
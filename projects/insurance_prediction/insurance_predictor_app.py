import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("Insurance Charges Prediction App")

st.write('Enter customer details to predict insurance charges')

age = st.slider('Age', 18, 100, 30)
sex = st.selectbox("Sex", ['female','male'])
bmi = st.slider('BMI', 10.0, 50.0, 25.0)
children = st.slider('Number of Children', 0, 10, 0)
smoker = st.selectbox("Smoker", ['yes','no'])
region = st.selectbox("Region", ['northeast', 'northwest', 'southeast', 'southwest'])

data = pd.read_csv('insurance.csv',index_col=0                                                                                                                                                                                                  )

X = data.drop(columns=['charges'])
y = data['charges']

X = pd.get_dummies(X, columns=['region'], drop_first=True, dtype=int)

X['sex'] = X['sex'].map({'female' : 1, 'male': 0})
X['smoker'] = X['smoker'].map({'yes': 1, 'no': 0})

X['age_smoker'] = X['age'] * X['smoker']
X['bmi_smoker'] = X['bmi'] * X['smoker']

model = LinearRegression()
model.fit(X,y)

input_data = pd.DataFrame({
    'age' : [age],
    'sex' : [1 if sex=='female' else 0],
    'bmi' : [bmi],
    'children' : [children],
    'smoker' : [1 if smoker == 'yes' else 0],
    'region_northwest' : [1 if region == 'northwest' else 0],
    'region_southeast' : [1 if region == 'southeast' else 0],
    'region_southwest' : [1 if region == 'southwest' else 0],
})

input_data['age_smoker'] = input_data['age'] * input_data['smoker']
input_data['bmi_smoker'] = input_data['bmi'] * input_data['smoker']

input_data = input_data[X.columns]

if st.button("Predict Charges"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated Insurance Charges : ${prediction:.2f}")
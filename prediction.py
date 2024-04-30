import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, OneHotEncoder

# Load the machine learning model and encodings/scalers
model = joblib.load('trained_model.pkl')
binary_encoding = joblib.load('gender_binary_encoding.pkl')
one_hot_encoding = joblib.load('geo_one_hot_encoder.pkl')
minmax_scaler = joblib.load('minmax_scaler.pkl')
robust_scaler = joblib.load('robust_scaler.pkl')

def main():
    st.title('Churn Model Deployment')
    
    HasCrCard = st.text_input("Surname: ")
    HasCrCard = st.radio("Geography: ", ["France","Germany", 'Spain'])
    CreditScore = st.number_input("Credit Score :", 300,900)
    Age = st.number_input("Input Age", 0, 100)
    Gender = st.radio("Input Gender : ", ["Male","Female"])
    Tenure = st.number_input("the period of time you holds a position (in years)", 0,100)
    Balance = st.number_input("Balance :")
    NumOfProducts = st.number_input("Number Of Products :")
    HasCrCard = st.radio("I Have a Credit Card : ", ["Yes","No"])
    IsActiveMember = st.radio("I am an Active Member : ", ["Yes","No"])
    EstimatedSalary = st.number_input("Estimated Salary :")

    
    data = {'CreditScore':int(CreditScore),
            'Gender': Gender, 'Age': int(Age), 
            'Tenure': int(Tenure), 'Balance': Balance,
            'NumOfProducts': int(NumOfProducts), 'HasCrCard': HasCrCard,
            'IsActiveMember':IsActiveMember,'EstimatedSalary': EstimatedSalary}
    
    df=pd.DataFrame([list(data.values())], columns=['CreditScore','Gender',  
                                                'Age', 'Tenure','Balance', 
                                                'NumOfProducts', 'HasCrCard' ,'IsActiveMember', 'EstimatedSalary'])
    
    # Replace categorical values with encoded values
    df=df.replace(binary_encoding)
    df = pd.get_dummies(df, columns=['Geography'])  # Apply one-hot encoding for Geography
    
    # Scale numerical features
    df['CreditScore'] = robust_scaler.transform(df[['CreditScore']])
    df['Age'] = robust_scaler.transform(df[['Age']])
    df['NumOfProducts'] = robust_scaler.transform(df[['NumOfProducts']])
    df['Tenure'] = minmax_scaler.transform(df[['Tenure']])
    df['Balance'] = minmax_scaler.transform(df[['Balance']])
    df['EstimatedSalary'] = minmax_scaler.transform(df[['EstimatedSalary']])
    
    if st.button('Make Prediction'):
        features=df      
        result = make_prediction(features)
        st.success(f'The prediction is: {result}')

def make_prediction(features):
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()

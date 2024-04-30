import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the machine learning model and encode
model = joblib.load('trained_model.pkl')
gender_encode= joblib.load('gender_binary_encoding.pkl')
geo_encode=joblib.load('geo_one_hot_encoder.pkl')
minmax_scaler= joblib.load('minmax_scaler.pkl')
robust_scaler=joblib.load('robust_scaler.pkl')

def main():
    st.title('Churn Model Deployment')

    # Add user input components for 10 features
    #input one by one
    id=st.number_input("id", 0, 200000)
    customer_id=st.number_input("Customer Id", 0, 15000000)
    surname = st.text_input("Surname")
    creditscore = st.number_input("Credit Score", 350, 850)
    geography=st.radio("Geography", ["France","Germany", "Spain"])
    gender=st.radio("Gender", ["Male","Female"])
    age=st.number_input("Age", 18, 92)
    tenure=st.number_input("Tenure", 0,10)
    balance=st.number_input("Balance", 0.0,251000.0)
    numproducts=st.radio("Number of Products", ["1","2", "3", "4"])
    creditcard=st.radio("Are you have a Credit Card? [0=No, 1=Yes]", ["0","1"])
    activeMem=st.radio("Are you an Active Member? [0=No, 1=Yes]", ["0","1"])
    estimatedSal=st.number_input("Estimated Salary", 11.580, 199992.48)


    data = {'Unnamed: 0': 0, 'id': int(id), 'CustomerId': int(customer_id), 'Surname': surname,
            'CreditScore': int(creditscore), 'Geography':geography, 'Gender':gender,
            'Age': int(age), 'Tenure':int(tenure), 'Balance': float(balance),
            'NumOfProducts':int(numproducts), 'HasCrCard': int(creditcard),
            'IsActiveMember':int(activeMem),'EstimatedSalary':float(estimatedSal)}

    df=pd.DataFrame([list(data.values())], columns=['Unnamed: 0', 'id', 'CustomerId', 'Surname', 'CreditScore', 'Geography',
                                                    'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
                                                    'IsActiveMember', 'EstimatedSalary'])

    df=df.replace(gender_encode)
    geography=df[['Geography']]
    geo_enc=pd.DataFrame(geo_encode.transform(geography).toarray(),columns=geo_encode.get_feature_names_out())
    df=pd.concat([df,geo_enc], axis=1)
    minmax_col = ['Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
    df[minmax_col] = minmax_scaler.transform(df[minmax_col])
    robust_col = ['CreditScore', 'Age']
    df[robust_col] = robust_scaler.transform(df[robust_col])
    df=df.drop(['Geography'],axis=1)
    df=df.drop('Unnamed: 0', axis=1)
    df=df.drop('id', axis=1)
    df=df.drop('CustomerId', axis=1)
    df=df.drop('Surname', axis=1)

    if st.button('Make Prediction'):
        features=df
        result = make_prediction(features)
        st.success(f'The prediction is: {result}')

def make_prediction(features):
    # Use the loaded model to make predictions
    # Replace this with the actual code for your model
    input_array = np.array(features).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction[0]

if __name__ == '__main__':
    main()

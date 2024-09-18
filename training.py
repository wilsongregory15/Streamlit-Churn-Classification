import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, RobustScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report
import pickle
import statistics

class DataHandler:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.input_df = None
        self.output_df = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path, delimiter=',')

    def create_input_output(self, target_column):
        self.output_df = self.data[target_column]
        self.input_df = self.data.drop(target_column, axis=1)
        self.input_df = self.data.drop('Unnamed: 0', axis=1)
        self.input_df = self.data.drop('id', axis=1)
        self.input_df = self.data.drop('CustomerId', axis=1)
        self.input_df = self.data.drop('Surname', axis=1)

class ModelHandler:
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.model = xgb.XGBClassifier()
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None] * 5

    def split_data(self, test_size=0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.input_data, self.output_data, test_size=test_size,
            random_state=random_state)

    def fill_missing_values(self):
        #Fill missing values for categorical columns with mode (Worst Case apabila data missing dalam data test)
        categorical_cols = ['Geography', 'Gender', 'HasCrCard', 'IsActiveMember']
        for col in categorical_cols:
            mode_val = statistics.mode(self.x_train[col])
            self.x_train[col].fillna(mode_val, inplace=True)
            self.x_test[col].fillna(mode_val, inplace=True)

        #Fill missing values for numerical columns with mean (Worst Case apabila ada missing value dalam data test)
        numerical_cols_mean = ['Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
        for col in numerical_cols_mean:
            mean_val = self.x_train[col].mean()
            self.x_train[col].fillna(mean_val, inplace=True)
            self.x_test[col].fillna(mean_val, inplace=True)

        numerical_cols_med = ['CreditScore', 'Age']
        for col in numerical_cols_med:
            median_val = self.x_train[col].median()
            self.x_train[col].fillna(median_val, inplace=True)
            self.x_test[col].fillna(median_val, inplace=True)
    
    def norm_and_standardize (self):
        #robust scaler untuk kolom dengan outliers
        robust_col = ['CreditScore', 'Age', 'NumOfProducts']
        robust_scaler = RobustScaler()
        self.x_train[robust_col] = robust_scaler.fit_transform(self.x_train[robust_col])
        self.x_test[robust_col] = robust_scaler.transform(self.x_test[robust_col])

        #minmax scaler untuk kolom tanpa outliers
        minmax_col = ['Tenure', 'Balance', 'EstimatedSalary']
        minmax_scaler = MinMaxScaler()
        self.x_train[minmax_col] = minmax_scaler.fit_transform(self.x_train[minmax_col])
        self.x_test[minmax_col] = minmax_scaler.transform(self.x_test[minmax_col])

        return robust_scaler, minmax_scaler

    def encode_categorical(self):
        #Binary Encoding untuk kolom 'Gender'
        gender_binary_encoding = {
            "Gender": {"Male": 1, "Female": 0},
        }
        self.x_train.replace(gender_binary_encoding, inplace=True)
        self.x_test.replace(gender_binary_encoding, inplace=True)

        #One Hot Encoding untuk kolom 'Geography'
        geo_one_hot_encoder = OneHotEncoder(drop=None)
        one_hot_cols = ['Geography']
        for col in one_hot_cols:
            train_encoded = geo_one_hot_encoder.fit_transform(self.x_train[[col]])
            test_encoded = geo_one_hot_encoder.transform(self.x_test[[col]])
            train_data = pd.DataFrame(train_encoded.toarray(), columns=geo_one_hot_encoder.get_feature_names_out([col]))
            test_data = pd.DataFrame(test_encoded.toarray(), columns=geo_one_hot_encoder.get_feature_names_out([col]))
            self.x_train = pd.concat([self.x_train.reset_index(), train_data], axis=1)
            self.x_test = pd.concat([self.x_test.reset_index(), test_data], axis=1)

        #Select only specified columns for x_train and x_test
        selected_cols = ['CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts',
                         'HasCrCard', 'IsActiveMember', 'EstimatedSalary', 'Geography_France',
                         'Geography_Spain', 'Geography_Germany']
        self.x_train = self.x_train[selected_cols]
        self.x_test = self.x_test[selected_cols]
        return geo_one_hot_encoder, gender_binary_encoding
    
    def save_one_hot_encoder(self, one_hot_encoder, filename):
        with open(filename, 'wb') as f:
            pickle.dump((one_hot_encoder), f)

    def save_binary_encoding(self, binary_encoding, filename):
        with open(filename, 'wb') as f:
            pickle.dump((binary_encoding), f)

    def save_minmax_scaler(self, minmax_scaler, filename):
        with open(filename, 'wb') as f:
            pickle.dump((minmax_scaler), f)

    def save_robust_scaler(self, robust_scaler, filename):
        with open(filename, 'wb') as f:
            pickle.dump((robust_scaler), f)

    def train_model(self):
        self.model.fit(self.x_train, self.y_train)

    def evaluate_model(self):
        self.y_predict = self.model.predict(self.x_test)
        return accuracy_score(self.y_test, self.y_predict)

    def make_report(self):
        print('\nClassification Report\n')
        print(classification_report(self.y_test, self.y_predict, target_names=['Class 0', 'Class 1']))

    def save_model(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.model, file)


#Load the data and define the input and output df
file_path = 'data_C.csv'
target_column = 'churn'
data_handler = DataHandler(file_path)
data_handler.load_data()
data_handler.create_input_output(target_column)
input_df = data_handler.input_df
output_df = data_handler.output_df

#Split data into training and testing, fill the missing values, encode, training and testing the model
model_handler = ModelHandler(input_df, output_df)
model_handler.split_data()
model_handler.fill_missing_values()
robust_scaler, minmax_scaler = model_handler.norm_and_standardize()
model_handler.save_minmax_scaler(minmax_scaler, 'minmax_scaler.pkl')
model_handler.save_robust_scaler(robust_scaler, 'robust_scaler.pkl')
geo_one_hot_encoder, gender_binary_encoding = model_handler.encode_categorical()
model_handler.save_one_hot_encoder(geo_one_hot_encoder, 'geo_one_hot_encoder.pkl')
model_handler.save_binary_encoding(gender_binary_encoding, 'gender_binary_encoding.pkl')
model_handler.train_model()
model_handler.evaluate_model()
print("Model Accuracy:", model_handler.evaluate_model())
model_handler.make_report()
model_handler.save_model('trained_model.pkl')

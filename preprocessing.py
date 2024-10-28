import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

directory = "D:\\Stat Project\\Datasets"
file_list = os.listdir(directory)
if __name__ == "__main__":
    print(file_list)

files_csv = []
for file_name in file_list:
    if file_name.endswith(".csv"):
        file_path = os.path.join(directory, file_name)
        files_csv.append(file_path)

def pre_process(files_csv=files_csv):
    processed_data = {}
    for dataset in files_csv:
        if dataset == 'D:\\Stat Project\\Datasets\\CAR DETAILS FROM CAR DEKHO.csv':
            car_details_file_path = dataset
            car_details_df = pd.read_csv(car_details_file_path)
            X = car_details_df.drop(columns=["name", "selling_price"])
            y = car_details_df["selling_price"]
            X = pd.get_dummies(X, drop_first=True)
            scaler = StandardScaler()
            numerical_features = ['year', 'km_driven']
            X[numerical_features] = scaler.fit_transform(X[numerical_features])
            X = np.c_[np.ones(X.shape[0]), X]  

            processed_data["CAR_DETAILS"] = (X, y) 
        elif dataset == 'D:\\Stat Project\\Datasets\\insurance.csv':
            insurance_file_path = dataset
            insurance_df = pd.read_csv(insurance_file_path)
            # insurance_df.head()
            X = insurance_df.drop(columns=["charges"]) 
            y = insurance_df["charges"]
            X = pd.get_dummies(X, drop_first=True)
            scaler = StandardScaler()
            numerical_features = ['age', 'bmi', 'children']
            X[numerical_features] = scaler.fit_transform(X[numerical_features])
            X = np.c_[np.ones(X.shape[0]), X] 
            processed_data["INSURANCE"] = (X, y) 

        elif dataset == 'D:\\Stat Project\\Datasets\\WHO COVID-19 cases.csv':
            covid_file_path = dataset
            covid_df = pd.read_csv(covid_file_path)
            covid_df = covid_df.drop(columns=["Date_reported", "Country_code", "Country", "WHO_region"])
            covid_df = covid_df.fillna(0)
            X = covid_df.drop(columns=["Cumulative_cases"])
            y = covid_df["Cumulative_cases"]
            X = pd.get_dummies(X, drop_first=True)
            scaler = StandardScaler()
            numerical_features = X.select_dtypes(include=['float64', 'int64']).columns
            X[numerical_features] = scaler.fit_transform(X[numerical_features])
            X = np.c_[np.ones(X.shape[0]), X]  
            X = np.array(X, dtype=float)
            y = np.array(y, dtype=float)
            processed_data["WHO_COVID"] = (X, y) 

    return processed_data

if __name__ == "__main__":
    processed_data = pre_process()
    print(processed_data)
    print("......................")
    print(processed_data["CAR_DETAILS"])
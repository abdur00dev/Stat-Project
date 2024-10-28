import numpy as np
import pandas as pd
from mle import theta_mle
from map import theta_map
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import preprocessing
from preprocessing import pre_process
import mle
import map
import other_model

processed_data = pre_process()
for dataset_name, (X, y) in processed_data.items():
    print(f"Linear regression with MLE & MAP on dataset '{dataset_name}':")
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # MLE model
    theta1 = mle.theta_mle(X_train, y_train)
    y_pred_mle = X_test @ theta1
    mse_mle = mean_squared_error(y_test, y_pred_mle)
    print(f"MSE of LR with MLE: {mse_mle}")



    # MAP model with lambda optimization
    lambda_values = np.linspace(0.01, 10.0, 100)
    best_lambda = None
    lowest_mse = float('inf')
    best_theta2 = None

    for lamda in lambda_values:
        theta2 = map.theta_map(X_train, y_train, lamda)
        y_pred_map = X_test @ theta2
        mse_map = mean_squared_error(y_test, y_pred_map)
        
        if mse_map < lowest_mse:
            lowest_mse = mse_map
            best_lambda = lamda
            best_theta2 = theta2
    print(f"Best Lambda: {best_lambda}")
    print(f"MSE of LR with MAP: {lowest_mse}")
    print("Other standard model MSE:")
    print(f"Linear model MSE:{other_model.func_linear_model(X_train,y_train,X_test,y_test)}")
    print(f"Ridge model MSE:{other_model.func_ridge_model(X_train,y_train,X_test,y_test)}")
    print(f"Random forest model MSE:{other_model.func_rf_model(X_train,y_train,X_test,y_test)}")
    print("...........................")
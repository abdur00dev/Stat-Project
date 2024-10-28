from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

linear_model = LinearRegression()
ridge_model = Ridge(alpha=0.5) 
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

def func_linear_model(X_train, y_train,X_test,y_test):
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)
    mse_linear = mean_squared_error(y_test, y_pred_linear)
    return mse_linear

def func_ridge_model(X_train, y_train,X_test,y_test):
    ridge_model.fit(X_train, y_train)
    y_pred_ridge = ridge_model.predict(X_test)
    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    return mse_ridge

def func_rf_model(X_train, y_train,X_test,y_test):
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    return mse_rf
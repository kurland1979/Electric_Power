from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import  MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error,root_mean_squared_error
import pandas as pd

def split_X_y(train,test):
    """
    Splits train and test DataFrames into features (X) and target (y) sets.
    Drops target and date columns (and 'naive_pred' if present) from X.

    Parameters:
        train (pd.DataFrame): Training set with features and target.
        test (pd.DataFrame): Test set with features and target.

    Returns:
        tuple: (X_train, y_train, X_test, y_test)
    """
   
    X_train = train.drop(columns=['Global_active_power', 'Date', 'naive_pred'], errors='ignore')
    y_train = train['Global_active_power']


    X_test = test.drop(columns=['Global_active_power', 'Date', 'naive_pred'], errors='ignore')
    y_test = test['Global_active_power']
    
    return X_train, y_train, X_test, y_test

def build_linear_regression(X_train, y_train):
    """
    Builds and fits a linear regression model with MinMaxScaler using a scikit-learn Pipeline.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        sklearn.pipeline.Pipeline: Fitted linear regression model pipeline.
    """
    
    model_linear = Pipeline([
        ('minmaxScaler',MinMaxScaler()),
        ('linearregression',LinearRegression())
    ])
    
    model_linear.fit(X_train,y_train)
    
    return model_linear

def set_linear(model_linear, X_test, y_test):
    """
    Evaluates the linear regression model on test data.
    Calculates MAE, MAE ratio, and RMSE for the predictions.

    Parameters:
        model_linear (sklearn.pipeline.Pipeline): Fitted linear regression pipeline.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True test target values.

    Returns:
        dict: Dictionary with 'mae_linear', 'mae_ratio_linear', and 'rmse_linear'.
    """
    
    y_pred = model_linear.predict(X_test)
    mae_linear = round(mean_absolute_error(y_test,y_pred),4)
    mae_ratio_linear = round(((mae_linear / y_test.mean()) * 100),4)
    rmse_linear = round(root_mean_squared_error(y_test,y_pred),4)
    
    result_linear = {
        'mae_linear': mae_linear,
        'mae_ratio_linear': mae_ratio_linear,
        'rmse_linear': rmse_linear
    }
    
    return result_linear

def build_linear_pipeline(train, test):
    """
    Pipeline for linear regression time series forecast.
    Returns: result metrics, fitted model
    """
    X_train, y_train, X_test, y_test = split_X_y(train, test)
    model_linear = build_linear_regression(X_train, y_train)
    result_linear = set_linear(model_linear, X_test, y_test)
    
    return result_linear, model_linear
    
    
    
    
 
    
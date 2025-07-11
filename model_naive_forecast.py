import pandas as pd
from sklearn.metrics import mean_absolute_error,root_mean_squared_error

def split_train_test(daily_df):
    """
    Splits the daily DataFrame into train and test sets (80/20 split).

    Parameters:
        daily_df (pd.DataFrame): DataFrame with daily aggregated features.

    Returns:
        tuple: (train, test) DataFrames.
    """
    
    total_size = int(len(daily_df) * 0.8)
    train = daily_df[:total_size]
    test = daily_df[total_size:]
    
    return train, test

def build_naive_forecast(train,test):
    """
    Builds a naive forecast for time series: predicts each day in test set as the previous day's actual value.
    For the first test day, uses the last train value.

    Parameters:
        train (pd.DataFrame): Training set with 'Global_active_power'.
        test (pd.DataFrame): Test set with 'Global_active_power'.

    Returns:
        pd.DataFrame: Test set with an added column 'naive_pred' containing naive predictions.
    """
    
    last_train_value = train['Global_active_power'].iloc[-1]
    test = test.copy()
    test['naive_pred'] = test['Global_active_power'].shift(1)
    test.loc[test.index[0], 'naive_pred'] = last_train_value

    return test

def set_naive_forecast(test):
    """
    Calculates evaluation metrics for the naive forecast: MAE, MAE ratio, and RMSE.

    Parameters:
        test (pd.DataFrame): Test set with true and naive predicted values.

    Returns:
        dict: Dictionary with 'mae_naive', 'mae_ratio_naive', and 'rmse_naive'.
    """
    
    mae_naive = mean_absolute_error(test['Global_active_power'],test['naive_pred']).round(4)
    mae_ratio_naive = ((mae_naive / test['Global_active_power'].mean()) * 100).round(4)
    rmse_naive = root_mean_squared_error(test['Global_active_power'],test['naive_pred']).round(4)
    
    result_naive = {
        'mae_naive': mae_naive,
        'mae_ratio_naive': mae_ratio_naive,
        'rmse_naive': rmse_naive
        }
    
    return result_naive
    
def build_naive_pipeline(daily_df):
    """
    Pipeline for naive time series forecast.
    Returns train, test (with predictions), and result metrics.
    """
    train, test = split_train_test(daily_df)
    if train is None or test is None: 
        return None
    test = build_naive_forecast(train, test)
    result_naive = set_naive_forecast(test)
    
    return train, test, result_naive   
    
    
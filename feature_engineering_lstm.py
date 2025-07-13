from sklearn.preprocessing import  MinMaxScaler
import config
import pandas as pd
import numpy as np


def split_train_val_test(daily_df):
    """
    Splits a DataFrame into train, validation, and test sets by chronological order.

    Parameters:
        daily_df (pd.DataFrame): The full daily DataFrame to be split.

    Returns:
        tuple: Three DataFrames (train, validation, test) preserving chronological order.
            - train_lstm (pd.DataFrame): The training set (first 70% of rows).
            - val_lstm (pd.DataFrame): The validation set (next 15% of rows).
            - test_lstm (pd.DataFrame): The test set (remaining rows).
    """
    
    total_size = len(daily_df)
    train_size = int(total_size * 0.7)
    val_size = int(total_size * 0.15)

    train_lstm = daily_df[:train_size]
    val_lstm = daily_df[train_size:train_size + val_size]
    test_lstm = daily_df[train_size + val_size:]

    return train_lstm, val_lstm, test_lstm

def build_features(train_lstm, val_lstm, test_lstm):
    """
    Selects features and target columns from train, validation, and test DataFrames.

    Parameters:
        train_lstm (pd.DataFrame): Training set DataFrame.
        val_lstm (pd.DataFrame): Validation set DataFrame.
        test_lstm (pd.DataFrame): Test set DataFrame.

    Returns:
        tuple:
            - train_X (pd.DataFrame): Features for training set.
            - train_y (pd.Series): Target for training set.
            - val_X (pd.DataFrame): Features for validation set.
            - val_y (pd.Series): Target for validation set.
            - test_X (pd.DataFrame): Features for test set.
            - test_y (pd.Series): Target for test set.
    """
    
    train_X = train_lstm[['month', 'dayofweek', 'season_encoded']]
    train_y = train_lstm['Global_active_power']

    val_X = val_lstm[['month', 'dayofweek', 'season_encoded']]
    val_y = val_lstm['Global_active_power']

    test_X = test_lstm[['month', 'dayofweek', 'season_encoded']]
    test_y = test_lstm['Global_active_power']

    
    return train_X,train_y,val_X,val_y,test_X,test_y

def scaler_feature(train_X,val_X,test_X):
    """
    Scales the feature sets using MinMaxScaler (fit on train, transform on all).

    Parameters:
        train_X (pd.DataFrame): Training features.
        val_X (pd.DataFrame): Validation features.
        test_X (pd.DataFrame): Test features.

    Returns:
        tuple:
            - train_X_scaled (np.ndarray): Scaled training features.
            - val_X_scaled (np.ndarray): Scaled validation features.
            - test_X_scaled (np.ndarray): Scaled test features.
            - scaler (MinMaxScaler): The fitted scaler object for future use.
    """
    
    scaler = MinMaxScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    val_X_scaled = scaler.transform(val_X)
    test_X_scaled = scaler.transform(test_X)
    
    
    return train_X_scaled,val_X_scaled,test_X_scaled, scaler 

def build_sequences(features, target, window_size=config.WINDOW_SIZE):
    target = target.reset_index(drop=True)
    """
    Generates sliding window sequences and corresponding targets for time series modeling.

    Parameters:
        features (np.ndarray or pd.DataFrame): Feature data for a single set (train/val/test), already scaled.
        target (np.ndarray or pd.Series): Target values for the same set.
        window_size (int): Number of time steps to include in each input sequence.

    Returns:
        tuple:
            - X (np.ndarray): 3D array of input sequences with shape (num_samples, window_size, num_features).
            - y (np.ndarray): 1D array of target values, one for each sequence.
    """
    
    X = []
    y = []
    for i in range(len(features) - window_size):
        window = features[i:i + window_size]
        target_value = target[i + window_size]
        X.append(window)
        y.append(target_value)
    X = np.array(X)
    y = np.array(y)
    return X, y

def build_sequences_pipeline(daily_df, window_size=config.WINDOW_SIZE):
    """
    Complete pipeline: splits data, selects features, scales, and generates time series sliding window sequences 
    for train, validation, and test sets.

    Parameters:
        daily_df (pd.DataFrame): The full daily DataFrame after cleaning and feature engineering.
        window_size (int): Number of time steps in each input sequence.

    Returns:
        tuple:
            - X_train_seq (np.ndarray): Sliding window sequences for the training set.
            - y_train_seq (np.ndarray): Target values for the training set.
            - X_val_seq (np.ndarray): Sliding window sequences for the validation set.
            - y_val_seq (np.ndarray): Target values for the validation set.
            - X_test_seq (np.ndarray): Sliding window sequences for the test set.
            - y_test_seq (np.ndarray): Target values for the test set.
            - scaler (MinMaxScaler): The fitted scaler object.
    """
    
    train_lstm, val_lstm, test_lstm = split_train_val_test(daily_df)
    train_X,train_y,val_X,val_y,test_X,test_y = build_features(train_lstm, val_lstm, test_lstm)

    train_X_scaled,val_X_scaled,test_X_scaled, scaler = scaler_feature(train_X,val_X,test_X)
    X_train_seq, y_train_seq = build_sequences(train_X_scaled, train_y, window_size)
    X_val_seq, y_val_seq = build_sequences(val_X_scaled, val_y, window_size)
    X_test_seq, y_test_seq = build_sequences(test_X_scaled, test_y, window_size)

    
    return X_train_seq, y_train_seq,X_val_seq,y_val_seq, X_test_seq, y_test_seq, scaler

    
    
     
    
    
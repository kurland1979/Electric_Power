from xgboost import XGBRegressor
from sklearn.preprocessing import  MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error,root_mean_squared_error
import pandas as pd

def build_xgboost( X_train, y_train):
    """
    Builds and fits a XGBRegressor model using a scikit-learn Pipeline with MinMaxScaler.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        sklearn.pipeline.Pipeline: Fitted XGBRegressor pipeline.
    """
    model_xgboost = Pipeline([
        ('minmaxScaler',MinMaxScaler()),
        ('xgbregressor',XGBRegressor())
    ])
    
    model_xgboost.fit(X_train,y_train)
    
    return model_xgboost

def set_xgboost(model_xgboost,X_test,y_test):
    """
    Evaluates the XGBRegressor model on test data.
    Calculates MAE, MAE ratio, and RMSE for the predictions.

    Parameters:
        model_xgboost(sklearn.pipeline.Pipeline): Fitted XGBRegressor pipeline.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True test target values.

    Returns:
        dict: Dictionary with 'mae_xgboost', 'mae_ratio_xgboost', and 'rmse_xgboost'.
    """
    y_pred = model_xgboost.predict(X_test)
   
    mae_xgboost = round(mean_absolute_error(y_test,y_pred),4)
    mae_ratio_xgboost = round(((mae_xgboost / y_test.mean()) * 100),4)
    rmse_xgboost = round(root_mean_squared_error(y_test,y_pred),4)
    
    result_xgboost = {
        'mae_xgboost': mae_xgboost,
        'mae_ratio_xgboost': mae_ratio_xgboost,
        'rmse_xgboost': rmse_xgboost
    }
    
    return  result_xgboost

def xgboost_feature_importances(model_xgboost,X_train):
    """
    Extracts and returns the feature importances from a fitted XGBRegressor model inside a pipeline.

    Parameters:
        model_xgboost  (sklearn.pipeline.Pipeline): 
            A trained pipeline containing a 'xgbregressor' step.
        X_train (pd.DataFrame): 
            The training DataFrame used for fitting the model, with feature column names.

    Returns:
        pd.DataFrame: 
            DataFrame sorted by feature importance (descending) with columns:
            - 'Feature': Feature names as in X_train
            - 'Importance': Corresponding importance values
    """
    
    importances = model_xgboost.named_steps['xgbregressor'].feature_importances_
    feature_names = X_train.columns
    xgb_features = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    return xgb_features


def build_xgboost_pipeline(X_train, y_train,X_test, y_test):
    """
    Pipeline for XGBRegressor time series forecast.
    Returns: result metrics, fitted model
    """
    
    model_xgboost = build_xgboost(X_train, y_train)
    result_xgboost = set_xgboost(model_xgboost, X_test, y_test)
    xgb_features = xgboost_feature_importances(model_xgboost,X_train)
    
    return result_xgboost, model_xgboost, xgb_features
    
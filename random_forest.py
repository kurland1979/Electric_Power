from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import  MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error,root_mean_squared_error
import pandas as pd

def build_random_forest( X_train, y_train):
    """
    Builds and fits a Random Forest regression model using a scikit-learn Pipeline with MinMaxScaler.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        sklearn.pipeline.Pipeline: Fitted Random Forest regression pipeline.
    """
    
    model_random_forest = Pipeline([
        ('minmaxScaler',MinMaxScaler()),
        ('randomforestregressor',RandomForestRegressor())
    ])
    
    model_random_forest.fit(X_train,y_train)
    
    return model_random_forest

def set_random_forest(model_random_forest, X_test, y_test):
    """
    Evaluates the Random Forest regression model on test data.
    Calculates MAE, MAE ratio, and RMSE for the predictions.

    Parameters:
        model_random (sklearn.pipeline.Pipeline): Fitted Random Forest regression pipeline.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True test target values.

    Returns:
        dict: Dictionary with 'mae_random', 'mae_ratio_random', and 'rmse_random'.
    """
    
    y_pred = model_random_forest.predict(X_test)
    mae_random = (mean_absolute_error(y_test,y_pred)).round(4)
    mae_ratio_random = ((mae_random / y_test.mean()) * 100).round(4)
    rmse_random = (root_mean_squared_error(y_test,y_pred)).round(4)
    
    result_random_forest = {
        'mae_random': mae_random,
        'mae_ratio_random': mae_ratio_random,
        'rmse_random': rmse_random
    }
    
    return result_random_forest

def random_feature_importances(model_random_forest,X_train):
    """
    Extracts and returns the feature importances from a fitted RandomForestRegressor model inside a pipeline.

    Parameters:
        model_random_forest (sklearn.pipeline.Pipeline): 
            A trained pipeline containing a 'randomforestregressor' step.
        X_train (pd.DataFrame): 
            The training DataFrame used for fitting the model, with feature column names.

    Returns:
        pd.DataFrame: 
            DataFrame sorted by feature importance (descending) with columns:
            - 'Feature': Feature names as in X_train
            - 'Importance': Corresponding importance values
    """
    
    importances = model_random_forest.named_steps['randomforestregressor'].feature_importances_
    feature_names = X_train.columns
    rf_features = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
        }).sort_values(by='Importance', ascending=False)
    
    return rf_features


def build_random_pipeline(X_train, y_train,X_test, y_test):
    """
    Pipeline for random regression time series forecast.
    Returns: result metrics, fitted model
    """
    
    model_random_forest = build_random_forest(X_train, y_train)
    result_random_forest = set_random_forest(model_random_forest, X_test, y_test)
    rf_features = random_feature_importances(model_random_forest,X_train)
    
    return result_random_forest, model_random_forest,rf_features
    
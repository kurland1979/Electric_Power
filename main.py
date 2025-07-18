from data_preprocessing import data_pipeline
from feature_engineering import consumption_pipeline
from model_naive_forecast import build_naive_pipeline
from linear_regression import split_X_y,build_linear_pipeline
from random_forest import build_random_pipeline
from model_xgboost import build_xgboost_pipeline
from visualization import build_plot_pipeline
from feature_engineering_lstm import build_sequences_pipeline
from dataloaders import get_dataloaders
from model_lstm import LSTMModel
from train_lstm import lstm_train,device
from evaluator import train_test
import os
import pickle
import config


def run_classical_models(daily_df):
    train, test, result_naive = build_naive_pipeline(daily_df)
    print('Metrics results Naive Forecast BaseLine:')
    print('*' * 40)
    print(f"MAE_NAIVE: {result_naive['mae_naive']}")
    print(f"MAE_RATIO_NAIVE: {result_naive['mae_ratio_naive']:.2f}%")
    print(f"RMSE_NAIVE: {result_naive['rmse_naive']}")
    print('='* 40)

    result_linear, model_linear = build_linear_pipeline(train, test)
    print()
    print('Metrics results LinearRegression:')
    print('*' * 40)
    print(f"MAE_LINEAR: {result_linear['mae_linear']:.4f}")
    print(f"MAE_RATIO_LINEAR: {result_linear['mae_ratio_linear']:.2f}%")
    print(f"RMSE_LINEAR: {result_linear['rmse_linear']:.4f}")

    X_train, y_train, X_test, y_test = split_X_y(train, test)
    print('='* 40)
    result_random_forest, model_random_forest, rf_features = build_random_pipeline(X_train, y_train, X_test, y_test)
    print()
    print('Metrics results RandomForestRegressor:')
    print('*' * 40)
    print(f"MAE_RANDOM_FOREST: {result_random_forest['mae_random']:.4f}")
    print(f"MAE_RATIO_RANDOM_FOREST: {result_random_forest['mae_ratio_random']:.2f}%")
    print(f"RMSE_RANDOM_FOREST: {result_random_forest['rmse_random']:.4f}")
    print('='* 40)

    result_xgboost, model_xgboost, xgb_features = build_xgboost_pipeline(X_train, y_train, X_test, y_test)
    print()
    print('Metrics results XGBRegressor:')
    print('*' * 40)
    print(f"MAE_XGBRegressor: {result_xgboost['mae_xgboost']:.4f}")
    print(f"MAE_RATIO_XGBRegressor: {result_xgboost['mae_ratio_xgboost']:.2f}%")
    print(f"RMSE_XGBRegressor: {result_xgboost['rmse_xgboost']:.4f}")
    print('='* 40)
    print()

    print(f'RANDOM_FEATURE:\n{rf_features}')
    print('*'* 40)
    print(f'XGB_FEATURE:\n{xgb_features}')

    build_plot_pipeline()
    results_without_MA = {
        'naive': {
            'MAE': result_naive['mae_naive'],
            'RMSE': result_naive['rmse_naive'],
            'MAE_RATIO': result_naive['mae_ratio_naive']
        },
        'linear_regression': {
            'MAE': result_linear['mae_linear'],
            'RMSE': result_linear['rmse_linear'],
            'MAE_RATIO': result_linear['mae_ratio_linear']
        },
        'random_forest': {
            'MAE': result_random_forest['mae_random'],
            'RMSE': result_random_forest['rmse_random'],
            'MAE_RATIO': result_random_forest['mae_ratio_random']
        },
        'xgboost': {
            'MAE': result_xgboost['mae_xgboost'],
            'RMSE': result_xgboost['rmse_xgboost'],
            'MAE_RATIO': result_xgboost['mae_ratio_xgboost']
        }
    }
    
    results_with_MA = {
        'naive': {
            'MAE': result_naive['mae_naive'],
            'RMSE': result_naive['rmse_naive'],
            'MAE_RATIO': result_naive['mae_ratio_naive']
        },
        'linear_regression': {
            'MAE': result_linear['mae_linear'],
            'RMSE': result_linear['rmse_linear'],
            'MAE_RATIO': result_linear['mae_ratio_linear']
        },
        'random_forest': {
            'MAE': result_random_forest['mae_random'],
            'RMSE': result_random_forest['rmse_random'],
            'MAE_RATIO': result_random_forest['mae_ratio_random']
        },
        'xgboost': {
            'MAE': result_xgboost['mae_xgboost'],
            'RMSE': result_xgboost['rmse_xgboost'],
            'MAE_RATIO': result_xgboost['mae_ratio_xgboost']
        }
    }
    
    
    return results_without_MA, results_with_MA, model_random_forest

def run_lstm(daily_df):
    X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq, scaler = build_sequences_pipeline(
        daily_df, window_size=config.WINDOW_SIZE)

    train_loader, val_loader, test_loader = get_dataloaders(
        X_train_seq, y_train_seq, X_val_seq, y_val_seq, X_test_seq, y_test_seq)

    model = LSTMModel()
    model.to(device)
    
    lstm_train(train_loader, val_loader, epochs=config.EPOCHS, patience=5)
    results = train_test(device, model, test_loader)
    print(f'MAE_LSTM: {results['mae']:.4f}')
    print(f'MSE_LSTM: {results['mse']:.4f}')
    print(f'R2_LSTM: {results['r2']:.4f}')
    print(f'RMSE_LSTM: {results['rmse']:.4f}')
    
"""
Main entry point for running the full time series forecasting project:
- Loads and preprocesses the raw data
- Engineers features (including rolling statistics)
- Trains and evaluates four models: Naive Forecast, Linear Regression, RandomForestRegressor, XGBRegressor
- Trains and evaluates LSTM model
- Saves results before and after rolling feature engineering
- Displays metrics, feature importances, and visualizations for model comparison

Usage:
    Run this script directly to execute the full workflow.
    Models and steps can be enabled/disabled using the RUN_* flags.

Modules imported:
    data_preprocessing, feature_engineering, model_naive_forecast, 
    linear_regression, random_forest, model_xgboost, etc.

Outputs:
    - Printed metrics for all models
    - Pickle files with results for further analysis and reporting
    - Visualization of model comparison and feature importance
"""  


if __name__ == '__main__':
    df = data_pipeline()
    daily_df = consumption_pipeline(df, verbose=False)

   
    results_without_MA, results_with_MA, model_random_forest = run_classical_models(daily_df)

   
    if os.path.exists('results_before_MA.pkl'):
        print("The file already exists, do not save again")
    else:
        with open('results_before_MA.pkl', 'wb') as f:
            pickle.dump(results_without_MA, f)
            print("Results saved!")

    if os.path.exists('results_after_MA.pkl'):
        print("The file already exists, do not save again")
    else:
        with open('results_after_MA.pkl', 'wb') as f:
            pickle.dump(results_with_MA, f)
            print("Results saved!")

   
    if os.path.exists('best_model.pkl'):
        print("The file already exists, do not save again")
    else:
        with open('best_model.pkl', 'wb') as f:
            pickle.dump(model_random_forest, f)

    
    print('\n', '='*10, 'LSTM Model', '='*10, '\n')
    
    run_lstm(daily_df)
    

  
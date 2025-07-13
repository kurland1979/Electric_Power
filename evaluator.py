import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        
def train_test(device,model,test_loader):
    """
    Evaluates the trained LSTM model on the test set and calculates regression metrics.

    Args:
        device (torch.device): The device on which to perform computation (CPU or CUDA).
        model (nn.Module): The trained LSTM model to evaluate.
        test_loader (DataLoader): DataLoader for the test set.

    Returns:
        dict: A dictionary containing test set evaluation metrics:
            - 'mae' (float): Mean Absolute Error.
            - 'mse' (float): Mean Squared Error.
            - 'r2' (float): R-squared (coefficient of determination).
            - 'rmse' (float): Root Mean Squared Error.
    """
    model.eval()
    all_predictions = []
    all_true_values = []
    with torch.no_grad():
        for X_test_batch, y_test_batch in test_loader: 
            X_test_batch = X_test_batch.to(device)
            y_test_batch = y_test_batch.to(device)
            outputs = model(X_test_batch)
                
            y_pred = outputs.squeeze().cpu().detach().numpy()
            y_true = y_test_batch.cpu().numpy()
            
            all_predictions.append(y_pred)
            all_true_values.append(y_true)
                
        all_predictions = np.concatenate(all_predictions)
        all_true_values = np.concatenate(all_true_values)
                
        mae = mean_absolute_error(all_true_values, all_predictions)
        mse = mean_squared_error(all_true_values, all_predictions)
        r2 = r2_score(all_true_values, all_predictions)
                
        result_test  = {
                        'mae': mae,
                        'mse': mse,
                        'r2': r2,
                        'rmse': np.sqrt(mse)
                        }  
            
        return result_test
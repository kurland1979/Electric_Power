import torch
import torch.nn as nn
from torch import optim
from model_lstm import LSTMModel
import config
import numpy as np


model = LSTMModel()
loss_fc = nn.MSELoss()
optimaizer = torch.optim.Adam(model.parameters(),lr=0.001)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def lstm_train(train_loader, val_loader, epochs=config.EPOCHS, patience=5):
    """
    Trains an LSTM model using the provided DataLoader objects for training and validation.
    Implements early stopping based on validation loss.

    Args:
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        epochs (int): Maximum number of training epochs.
        patience (int): Number of epochs with no improvement after which training will be stopped early.

    Returns:
        None
    """
    
    best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(epochs):
        all_loss_train = []
        for X_train_batch, y_train_batch in train_loader: 
            X_train_batch = X_train_batch.to(device)
            y_train_batch = y_train_batch.to(device)
        if y_train_batch.dim() == 1:
            y_train_batch = y_train_batch.unsqueeze(1)
            outputs = model(X_train_batch)
            loss_train = loss_fc(outputs, y_train_batch)
            optimaizer.zero_grad()
            loss_train.backward()
            optimaizer.step()
            all_loss_train.append(loss_train.item())
            
        mean_batch_train = sum(all_loss_train) / len(all_loss_train)
        loss_val = train_val(val_loader, model, loss_fc, device)
        # early stopping
        if loss_val < best_val_loss:
            best_val_loss = loss_val
            patience_counter = 0
        else:
            patience_counter += 1
                
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        print(f'Epoch {epoch+1}: Loss_Train: {mean_batch_train:.4f}, Loss_Val: {loss_val:.4f}')


def train_val(val_loader, model, loss_fc, device):
    """
    Evaluates the given model on a validation set and computes the average loss.

    Args:
        val_loader (DataLoader): DataLoader for the validation set.
        model (nn.Module): The trained LSTM model to evaluate.
        loss_fc (Loss): Loss function to use for evaluation.
        device (torch.device): The device on which to perform computation.

    Returns:
        float: Mean loss across all batches in the validation set.
    """
    
    model.eval()
    all_loss_val = []
    all_true_values = []
    for X_val_batch, y_val_batch in val_loader:
        X_val_batch = X_val_batch.to(device)
        y_val_batch = y_val_batch.to(device)
        if y_val_batch.dim() == 1:
            y_val_batch = y_val_batch.unsqueeze(1)
        outputs = model(X_val_batch)
        loss_val = loss_fc(outputs, y_val_batch)
        all_loss_val.append(loss_val.item())
        all_true_values.append(y_val_batch.cpu().numpy())
    if all_true_values:
        all_true_values = np.concatenate(all_true_values)
    mean_batch_val = sum(all_loss_val) / len(all_loss_val)
    model.train()
    
    return mean_batch_val




import torch.nn as nn
import config

class LSTMModel(nn.Module):
    """
    LSTM-based neural network for time series regression.

    Parameters:
        input_dim (int): Number of features in each time step.
        hidden_dim (int): Number of units in the LSTM hidden state.
        num_layers (int): Number of stacked LSTM layers.
        output_dim (int): Number of output units (usually 1 for regression).
        dropout (float): Dropout rate between LSTM layers.

    Forward Input:
        x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, input_dim).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, output_dim).
    """
    
    def __init__(self, input_size=config.INPUT_SIZE,
                 hidden_size=config.HIDDEN_SIZE,
                 num_layers=config.NUM_LAYERS, dropout=config.DROPOUT, output_size=config.OUTPUT_SIZE):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.bn = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        lstm_out, _ = self.lstm(x)           
        last_output = lstm_out[:, -1, :]       
        normed_output = self.bn(last_output)  
        drop = self.dropout(normed_output)     
        out = self.fc(drop)                   

        return out

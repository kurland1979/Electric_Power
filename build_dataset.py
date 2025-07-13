import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    """
    Custom PyTorch Dataset for time series sliding window sequences.

    Each sample contains one input sequence (X) and its corresponding target value (y).

    Parameters:
        X (np.ndarray or torch.Tensor): Input sequences, shape (num_samples, window_size, num_features).
        y (np.ndarray or torch.Tensor): Target values, shape (num_samples,) or (num_samples, 1).

    Methods:
        __len__: Returns the number of samples in the dataset.
        __getitem__: Returns the (X, y) pair at a given index as torch.Tensor objects.
    """
    
    def __init__(self, X, y):
        self.X = torch.Tensor(X)
        self.y = torch.Tensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index] 

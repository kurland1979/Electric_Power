from torch.utils.data import DataLoader
from build_dataset import SequenceDataset  
import config

def get_dataloaders(X_train_seq, y_train_seq,X_val_seq,y_val_seq, X_test_seq, y_test_seq, batch_size=config.BATCH_SIZE):
    """
    Creates PyTorch DataLoaders for train, validation, and test sets, using sliding window sequences.

    Parameters:
        X_train_seq (np.ndarray): Input sequences for training.
        y_train_seq (np.ndarray): Targets for training.
        X_val_seq (np.ndarray): Input sequences for validation.
        y_val_seq (np.ndarray): Targets for validation.
        X_test_seq (np.ndarray): Input sequences for test.
        y_test_seq (np.ndarray): Targets for test.
        batch_size (int): Number of samples per batch in each DataLoader.

    Returns:
        tuple:
            - train_loader (DataLoader): DataLoader for training set (shuffle=False).
            - val_loader (DataLoader): DataLoader for validation set (shuffle=False).
            - test_loader (DataLoader): DataLoader for test set (shuffle=False).
    """
    
    train_dataset = SequenceDataset(X_train_seq, y_train_seq)
    val_dataset = SequenceDataset(X_val_seq,y_val_seq)
    test_dataset = SequenceDataset(X_test_seq, y_test_seq)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader,val_loader, test_loader

import pandas as pd
from torch.utils.data import Dataset

import torch


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = pd.DataFrame(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X_train_tensor = torch.tensor(self.X.iloc[idx, :].values)
        y_train_tensor = torch.tensor(self.y.iloc[idx].values)

        return X_train_tensor, y_train_tensor

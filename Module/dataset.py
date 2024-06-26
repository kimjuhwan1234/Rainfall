import torch
import pandas as pd
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.data = pd.concat([X, y], axis=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_tensor = torch.tensor(self.data.iloc[idx, :].values)

        return data_tensor

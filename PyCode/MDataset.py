import numpy as np
import torch
from torch.utils.data import Dataset

class MDataset(Dataset):
    def __init__(self, x, y):
        self.data = x
        self.labels = y.astype(int)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = torch.tensor(self.data[idx])
        label = torch.tensor(self.labels[idx], dtype=int)
        
        return data, label
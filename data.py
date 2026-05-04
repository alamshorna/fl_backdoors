import torch
from torch.utils.data import Dataset


class PoisonedDataset(Dataset):
    def __init__(self, dataset, target=0, frac=0.1):
        self.dataset = dataset
        self.target = target
        self.n = int(len(dataset) * frac)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        x, y = self.dataset[i]
        if i < self.n:
            x = x.clone()
            x[:, -3:, -3:] = 1.0
            y = self.target
        return x, y

import torch
import numpy as np
from torch.utils.data import Dataset


class TrafficVolumeDataset(Dataset):
    def __init__(self, traffic_volume_data, time_step=24):
        self.traffic_volume_data = traffic_volume_data
        self.time_step = time_step

        self.X, self.y = self.create_dataset(self.traffic_volume_data, time_step)

    def create_dataset(self, data, time_step=24):
        X, y = [], []
        for i in range(len(data) - time_step - 1):
            X.append(data[i : (i + time_step)])
            y.append(data[i + time_step])
        return np.array(X), np.array(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]).view(-1, 1), torch.FloatTensor(
            [self.y[idx]]
        )
        

class TrafficVolumeDatasetMulti(Dataset):
    def __init__(self, features, targets, time_step=24):
        self.features = features
        self.targets = targets
        self.time_step = time_step

        self.X, self.y = self.create_dataset(self.features, self.targets, time_step)

    def create_dataset(self, features, targets, time_step=24):
        X, y = [], []
        for i in range(len(features) - time_step):
            X.append(features[i : (i + time_step)])
            y.append(targets[i + time_step])
        return np.array(X), np.array(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.X[idx]), torch.FloatTensor([self.y[idx]])

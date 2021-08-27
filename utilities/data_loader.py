import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class cnnLSTMDataset():

    def __init__(self,
                 X,
                 pid=[],
                 y=[],
                 transform=None,
                 target_transform=None):
        """
        Y needs to be in one-hot encoding
        X needs to be in N * Width
        Pid is a numpy array of size N
        Args:
            data_path (string): path to data
            files_to_load (list): subject names
            currently all npz format should allow support multiple ext

        """

        self.X = torch.from_numpy(X)
        self.y = y
        self.pid = pid
        self.unique_pid_list = np.unique(pid)
        self.transform = transform
        self.targetTransform = target_transform
        print("Total sample count : " + str(len(self.X)))

    def __len__(self):
        return len(self.unique_pid_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pid_of_choice = self.unique_pid_list[idx]
        sample_filter = self.pid == pid_of_choice
        sample = self.X[sample_filter, :]

        y = self.y[sample_filter]
        if self.targetTransform:
            y = [self.targetTransform(ele) for ele in y]
            # y = self.targetTransform(y)

        if self.transform:
            sample = self.transform(sample)
        return sample, y, self.pid[sample_filter]


class SubjectDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, pid, batch_size=2000):
        """
        batch_size: number of epochs from one subject
        """
        self.x = x
        self.y = y
        self.pid = pid
        self.batch_size = batch_size

    def __len__(self):
        return int(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        x = self.x[start_idx:start_idx+self.batch_size]
        y = self.y[start_idx:start_idx+self.batch_size]
        pid = self.pid[start_idx:start_idx+self.batch_size]

        return x, y, pid


class NormalDataset:

    def __init__(self,
                 X,
                 y=[],
                 name="",
                 isLabel=False,
                 transform=None,
                 target_transform=None):
        """
        Y needs to be in one-hot encoding
        X needs to be in N * Width
        Args:
            data_path (string): path to data
            files_to_load (list): subject names
            currently all npz format should allow support multiple ext

        """

        self.X = torch.from_numpy(X)
        self.y = y
        self.isLabel = isLabel
        self.transform = transform
        self.targetTransform = target_transform
        print(name+" set sample count : " + str(len(self.X)))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.X[idx, :]
        y = []
        if (self.isLabel):
            y = self.y[idx]
            if self.targetTransform:
                y = self.targetTransform(y)

        if self.transform:
            sample = self.transform(sample)
        return sample, y

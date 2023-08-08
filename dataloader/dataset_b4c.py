import os
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader

class B4CDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_list = [f for f in os.listdir(data_dir) if f.endswith('.npy') or f.endswith('.pkl')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.data_dir, file_name)

        if file_name.endswith('.npy'):
            data = np.load(file_path)
        elif file_name.endswith('.pkl'):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        else:
            raise ValueError(f"Unsupported file type: {file_name}")

        return data


from torch.utils.data import Dataset, DataLoader
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, data, len_dataset):

        self.data = data
        self.len_dataset = len_dataset

    def __len__(self):
        """Return the total number of samples in the dataset."""
        game_num = np.random.randint(0, self.len_dataset)
        game = None

        move_num = np.random.randint(0, len(game))

        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve a single data point and its label based on the index.

        Args:
            idx: Index of the data point to retrieve.

        Returns:
            A tuple (data, label) for the given index.
        """
        return self.data[idx], self.labels[idx]
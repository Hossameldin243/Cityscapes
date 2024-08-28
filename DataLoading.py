# Dataset Class

from torch.utils.data import DataLoader, Dataset
from typing import List
from torchvision.transforms import Compose
import pandas as pd
import os


class dataset(Dataset):
    def __init__(self, input_dataframe: pd.DataFrame, root_dir: str, KeysOfInterest: List[str], data_transform:Compose):
        self.root_dir = root_dir
        self.koi = KeysOfInterest
        self.input_dataframe = input_dataframe[self.koi]
        self.data_transform=data_transform

    def __getitem__(self, idx):
        sample = {}
        for key in self.koi:
            file = self.input_dataframe.iloc[idx][key]
            path = os.path.join(self.root_dir, file)
            sample[key] = path
            
        if self.data_transform:
            sample = self.data_transform(sample)
        
        return sample
    
    def __len__(self):
        return len(self.input_dataframe)
    
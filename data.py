import pandas as pd
import torch
from torch.utils.data.dataloader import Dataset
import soundfile as sf
import os


def handle_bad_samples_collate_fn(batch):
    # Filter out None values (missing samples)
    batch = [item for item in batch if item[0] is not None]
    if not batch:
        return None 
    tensors = [item[0] for item in batch]
    labels = torch.tensor([item[1] for item in batch])
    return torch.stack(tensors), labels


class PrepAudioDataset(Dataset):
    def __init__(self, 
        root_path, 
        label_path, 
        label_column_name: str, 
        filename_column_name: str,
        real_label_name='real',
        fake_label_name='fake',
        **kwargs):
        self.root_path = root_path 
        self.label_path = label_path 
        self.labels_df = self.load_all_labels()
        self.label_column_name = label_column_name 
        self.filename_column_name = filename_column_name 
        self.real_label, self.fake_label = real_label_name, fake_label_name

    def load_all_labels(self):
        df = pd.read_csv(self.label_path, sep=" ")
        return df

    def __len__(self):
        return self.labels_df.shape[0] 

    def __getitem__(self, index):
        filename = self.labels_df.loc[index, self.filename_column_name]
        label = self.labels_df.loc[index, self.label_column_name] 
        label = self.label_encode(label)
        data_path = self.root_path.rstrip("/") + "/" + filename 
        if os.path.exists(data_path):
            sample, _ = sf.read(data_path)
            sample = torch.tensor(sample, dtype=torch.float32)
        else:
            return None, label
        sample = torch.unsqueeze(sample, 0)
        return sample, label 

    
    def get_weights(self):
        label_info = self.labels_df.loc[:, self.label_column_name]
        num_zero_class = (label_info == self.real_label).sum()
        num_one_class = (label_info == self.fake_label).sum()
        weights = torch.tensor([num_one_class, num_zero_class], dtype=torch.float32)
        weights = weights / (weights.sum())
        return weights

    def label_encode(self, label):
        encoding = {
            self.fake_label: 1,
            self.real_label: 0  
        }
        return encoding[label]

    def label_decode(self, integer_label):
        encoding = {
            0: self.real_label,
            1: self.fake_label
        }
        return encoding[integer_label]

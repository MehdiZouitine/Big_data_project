import pandas as pd
from torch.utils.data import Dataset, Sampler
import torch
from transformers import BertTokenizerFast


class NlpTrainDataset(Dataset):
    #dataset
    def __init__(self, df, labels, tokenizer):

        self.labels = labels.tolist()
        self.encodings = tokenizer.batch_encode_plus(
            df["description"].to_list(), max_length=170, padding=True, truncation=True
        )
        self.Id = df["Id"].to_list()
        self.gender = df["gender"].to_list()

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        item["Id"] = self.Id[idx]
        item["gender"] = torch.tensor([self.gender[idx]], dtype=torch.float32)
        return item

    def __len__(self):
        return len(self.Id)


class NlpTestDataset(Dataset):
    def __init__(self, df, tokenizer):

        self.encodings = tokenizer.batch_encode_plus(
            df["description"].to_list(), max_length=170, padding=True, truncation=True
        )
        self.Id = df["Id"].to_list()
        self.gender = df["gender"].to_list()

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["Id"] = self.Id[idx]
        item["gender"] = torch.tensor([self.gender[idx]], dtype=torch.float32)
        return item

    def __len__(self):
        return len(self.Id)


class MetricLearningTrainDataset(Dataset):
    def __init__(self, df, labels, tokenizer):

        self.labels = labels
        self.encodings = tokenizer.batch_encode_plus(
            df["description"].to_list(), max_length=170, padding=True, truncation=True
        )
        self.Id = df["Id"]
        self.gender = df["gender"]

    def _get_sample(self, idx):
        sample = {
            "item": None,
            "positive_male": None,
            "positive_female": None,
            "negative": None,
        }
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx].values)
        item["Id"] = self.Id[idx]
        item["gender"] = torch.tensor([self.gender[idx]], dtype=torch.float32)
        self.labels.loc

    def __getitem__(self, idx):
        sample = item
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        item["Id"] = self.Id[idx]
        item["gender"] = torch.tensor([self.gender[idx]], dtype=torch.float32)

        positive
        return item

    def __len__(self):
        return len(self.Id)
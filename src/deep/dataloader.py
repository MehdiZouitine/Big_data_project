import pandas as pd
from torch.utils.data import Dataset
import torch
from transformers import BertTokenizerFast
import numpy as np
import pandas


class NlpTrainDataset(Dataset):
    def __init__(
        self,
        df: pandas.core.frame.DataFrame,
        labels: pandas.core.frame.DataFrame,
        tokenizer: BertTokenizerFast,
    ):
        """[Dataloader to generate shuffle batch of data]

        Args:
            df (pandas.core.frame.DataFrame): [Train text]
            labels (pandas.core.frame.DataFrame): [Train labels (jobs)]
            tokenizer (BertTokenizerFast): [Text tokenizer for bert]
        """

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
    def __init__(
        self, df: pandas.core.frame.DataFrame, tokenizer: pandas.core.frame.DataFrame
    ):
        """[Dataloader to generate test batch of data]

        Args:
            df (pandas.core.frame.DataFrame): [Train text]
            tokenizer (BertTokenizerFast): [Text tokenizer for bert]
        """

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

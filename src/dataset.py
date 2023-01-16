from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
import pandas as pd
import torch
from typing import Dict
import os


class MyDataset(Dataset):

    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self._df = df

    def __len__(self):
        return self._df.shape[0]

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, torch.Tensor]:
        return dict(self._df.iloc[index])


class DataModule(pl.LightningDataModule):

    def __init__(
        self,
        train_path: str,
        test_path: str,
        batch_size: int = 64,
        seed: int = 42,
    ):
        super().__init__()

        train_df = pd.read_pickle(train_path)
        test_df = pd.read_pickle(test_path)

        self.df = {
            "train": train_df,
            "test": test_df,
            "all": pd.concat([train_df, test_df]),
        }
        self.batch_size = batch_size

    def train_dataloader(self) -> DataLoader:
        return self._dataloader("train")

    def val_dataloader(self) -> DataLoader:
        return self._dataloader("test")

    def all_dataloader(self) -> DataLoader:
        return self._dataloader("all")

    def _dataloader(self, split: str) -> DataLoader:
        return DataLoader(
            MyDataset(self.df[split]),
            batch_size=self.batch_size,
            shuffle=split == "train",
            num_workers=int(os.environ.get("NUM_WORKERS", 0)),
        )

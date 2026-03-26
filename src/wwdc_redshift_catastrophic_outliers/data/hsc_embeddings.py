import os

import numpy as np 
import lightning as L
import torch
from datasets import Dataset as HFDataset
from datasets import load_from_disk
from torch.utils.data import DataLoader, Dataset

from dotenv import load_dotenv

from wwdc_redshift_catastrophic_outliers.data.modules import merge_datasets
load_dotenv() 

DATA_ROOT = os.getenv("DATA_ROOT")

bands = ["g", "r", "i", "z", "y"]

class EmbeddingDataset(Dataset):
    def __init__(
        self, 
        split,
        y_catalog=None,
    ):
        """
        Args:
            embedding_dir (str): The directory containing the embeddings.
        """
        self.dset = merge_datasets(
            split=split
        )
        self.y_catalog = y_catalog
    
    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        item = self.dset[idx]

        y = torch.stack(
            [
            torch.as_tensor(item[k])
            for k in self.y_catalog["variables"]
            if k not in self.y_catalog["drop_variables"]
            ]
        )

        X = torch.as_tensor(item["h"])
        z = torch.as_tensor(item["specz_redshift"])
        return X, y, z

class EmbeddingDataLoader(L.LightningDataModule):
    def __init__(
        self,
        datasets,
        batch_size=64,
        random_state=42,
        shuffle=True,
        num_workers=None,
    ):
        super().__init__()
        self.datasets = datasets
        self.batch_size = batch_size
        self.random_state = random_state
        if not num_workers:
            num_workers = os.cpu_count() - 1
            if num_workers > 16:  # limit worker allocation to stop
                # torch dataloader complaints. Need better way of doing this.
                self.num_workers = 16
            else:
                self.num_workers = num_workers
        else:
            self.num_workers = num_workers
        self.shuffle = shuffle

    def setup(self, _stage=None):
        self.train_dataset = self.datasets["train"]
        self.val_dataset = self.datasets["val"]
        self.test_dataset = self.datasets["test"]

    def base_dataloader(self, dataset, split):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle if split == "train" else False,
            num_workers=self.num_workers,
        )

    def train_dataloader(self):
        return self.base_dataloader(self.train_dataset, "train")

    def val_dataloader(self):
        return self.base_dataloader(self.val_dataset, "val")

    def test_dataloader(self):
        return self.base_dataloader(self.test_dataset, "test")

if __name__ == "__main__":
    y_catalog = {
        "catalog_name": "hsc_embed",
        "fp": None,
        "join_method": None,
        "variables": {
            "g_cmodel_mag": {
                "name": "g mag",
                "size": 1,
                "processing_fn": None,
            },
            "r_cmodel_mag": {
                "name": "r mag",
                "size": 1,
                "processing_fn": None,
            },
            "i_cmodel_mag": {
                "name": "i mag",
                "size": 1,
                "processing_fn": None,
            },
            "z_cmodel_mag": {
                "name": "z mag",
                "size": 1,
                "processing_fn": None,
            },
            "y_cmodel_mag": {
                "name": "y mag",
                "size": 1,
                "processing_fn": None,
            }
        },
        "drop_variables": [],
    }
    
    dset_dict = {
        "train": EmbeddingDataset(
            split="train",
            y_catalog=y_catalog
        ),
        "val": EmbeddingDataset(
            split="val",
            y_catalog=y_catalog
        ),
        "test": EmbeddingDataset(
            split="test",
            y_catalog=y_catalog
        )
    }
    dataloader = EmbeddingDataLoader(
        datasets=dset_dict
    )
    print(dataloader)
    dataloader.setup()
    df = dataloader.test_dataset

    batch = next(iter(df))

    X, y, z = batch
    print(X.shape, y.shape, z.shape)
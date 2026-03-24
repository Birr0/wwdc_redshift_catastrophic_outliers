import os

import pandas as pd 
import h5py
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv

load_dotenv() 

HF_USERNAME = os.getenv("HF_USERNAME")
HF_TOKEN = os.getenv("HF_TOKEN")
DATA_ROOT =  os.getenv("DATA_ROOT")

splits = ["test", "val", "train"]

def build_split(h5_path, split_name):
    def gen():
        with h5py.File(h5_path, "r") as f:
            h = f["h"]
            for i in range(len(h)):
                yield {
                    "h": h[i].tolist() if hasattr(h[i], "tolist") else h[i],
                }
    return Dataset.from_generator(gen)

def upload():
    ds_z = load_dataset(
        "csv", 
        data_files={
            split: f"{DATA_ROOT}/wwdc_catastrophic_z/metadata/{split}_pred_vs_true.csv"
            for split in splits
        }
    )
    ds_z.save_to_disk(
        f"{DATA_ROOT}/wwdc_catastrophic_z/metadata"
    )
    ds_z.push_to_hub(f"{HF_USERNAME}/hsc_pred_v_true")
    
    ds_meta = load_dataset(
        "csv", 
        data_files={
            split: f"{DATA_ROOT}/GalaxiesML/metadata/5x127x127_{split}_with_morphology.csv"
            for split in splits
        }
    )
    ds_meta.save_to_disk(
        f"{DATA_ROOT}/GalaxiesML/metadata"
    )
    ds_meta.push_to_hub(f"{HF_USERNAME}/hsc_metadata")

    data = {}
    for split in splits:
        data[split] = build_split(
            f"{DATA_ROOT}/wwdc_catastrophic_z/embeddings_raw/{split}_embeddings.h5",
            split
        )
    ds_embed = DatasetDict(data)
    ds_embed.save_to_disk(
        f"{DATA_ROOT}/wwdc_catastrophic_z/embeddings"
    )
    ds_embed.push_to_hub(f"{HF_USERNAME}/hsc_z_embedding")

if __name__ == "__main__":
    upload()
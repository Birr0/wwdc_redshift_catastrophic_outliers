import os 

from datasets import load_from_disk
from dotenv import load_dotenv
import numpy as np 

load_dotenv() 

DATA_ROOT = os.getenv("DATA_ROOT")

def merge_datasets(split):
    '''
    Implement cut on i_cmodel_mag <= 22.0
    '''

    ds_z = load_from_disk(
        f"{DATA_ROOT}/wwdc_catastrophic_z/metadata"
    )[split]
    ds_embed = load_from_disk(
        f"{DATA_ROOT}/wwdc_catastrophic_z/embeddings"
    )[split]
    ds_hsc = load_from_disk(
        f"{DATA_ROOT}/GalaxiesML/metadata"
    )[split]
    
    i_mag = np.array(ds_hsc["i_cmodel_mag"])
    idx = np.where(i_mag <= 22)[0] # hard-code mag cut.
    filtered_dset = ds_hsc.select(idx)

    for col in ds_z.column_names:
        filtered_dset = filtered_dset.add_column(f"{col}", ds_z[col])
    
    for col in ds_embed.column_names:
        filtered_dset = filtered_dset.add_column(f"{col}", ds_embed[col])
    
    return filtered_dset


if __name__ == "__main__":
    dset = merge_datasets(split="test")
    print(dset)

    print(dset.column_names)

    '''z = dset["specz_redshift"]
    print(z[:5])

    z_other = dset["y_0"]
    print(z_other[:5])'''


# WWDC: catastrophic redshift outliers

The goal of this project is to use the conditional embeddings to examine catastrophic redshift outliers from HSC images.

## Virtual enviroment
We use UV to manage installations

A simple uv sync should install the necessary dependencies. More descriptive docs are needed here...

## Training
Ensure the enviroment is active. 

```
srun python train.py -cn "experiment/hsc_conditional_flow/train" hydra/launcher={cluster_config_name}
```

## Embeddings
Ensure the enviroment is active. 

```
srun python embed.py -cn "experiment/hsc_conditional_flow/embed" hydra/launcher={cluster_config_name}
```

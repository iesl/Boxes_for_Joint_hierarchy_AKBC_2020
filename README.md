# Boxes_for_Joint_hierarchy_AKBC_2020

## Setup

1. Clone the repo

2. Install the requirements:

```
pip install -r requirements.txt
```

### Getting the preprocessed data

The data used in the paper can be found [here](https://drive.google.com/file/d/1LA_WOhQ9NnxmbXTuRx-Yhqtwpkk5dJmv/view?usp=sharing).

### Reproducing the results for box embeddings

#### Without Weights&Biases logging

```
export DATA_DIR=directory/where/unzipped/data/folder/is
export CUDA_DEVICE=0  # =-1 for cpu
export WANDB=false 
allennlp train model_configs/hypernym_0.jsonnet --serialization-dir hypernym_0_training_dump --include-package=datasets --include-package=boxes --include-package=model
```

#### With Weights&Biases logging

Assuming your username and project is `username` and `project` respectively.
```
export DATA_DIR=directory/where/unzipped/data/folder/is
export CUDA_DEVICE=0  # =-1 for cpu
export WANDB=false 
wandb_allennlp --subcommand=train --config_file=model_configs/hypernym_0.jsonnet --include-package=datasets --include-package=boxes --include-package=models --wandb_entity=username --wandb_project=project --wandb_run_name=hypernym_0
```


### Using the trained models on test data

```
export DATA_DIR=directory/where/unzipped/data/folder/is
python predict_f1_test.py --model hypernym_0_training_dump 
```

Replace `hypernym_0` with `hypernym_{10,25,50}`, `meronym_{0,10,25,30}` and `joint` to train all the regularized box models reported in the paper.



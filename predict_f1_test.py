from typing import Dict, List, Iterator, Union, Any
import logging
import tempfile
import json
import shutil
import os
from pathlib import Path
from pprint import pprint
import torch, tqdm
from models.utils.allennlp import (
    load_config, load_dataset_reader, load_iterator, load_model,load_best_metrics, load_modules, 
    load_outputs, create_onepass_generator)
from allennlp.models import Model
from allennlp.data import DatasetReader, DataIterator
import pandas as pd
from copy import deepcopy
import numpy as np
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model', dest = "model_dir", required=True, help='Path to the directory containing the saved model.')
parser.add_argument('--data', dest = "data_dir", help='Path to the directory containing all the data directories.')
args = parser.parse_args()
MODEL_DIR = args.model_dir
breakpoint()
DATA_DIR = args.data_dir or os.environ.get('DATA_DIR', None)

if not DATA_DIR:
	raise ValueError("Either set environ variable DATA_DIR or pass --data as arg to the script.")

# load modules
EXTRA_MODULES = ['models', 'datasets']
logger.info("Loading {} for AllenNLP".format(EXTRA_MODULES))
load_modules(EXTRA_MODULES)

# override the val dataset setup to become the test dataset setup
overrides = {'validation_dataset_reader':
             {
                 'all_datadir': DATA_DIR,
                 "validation_file": "classification_samples_test2id.txt"
             }
            }
logger.info("Setting up config overrides ...")

# load config
config = load_config(
   MODEL_DIR, overrides_dict=overrides)
#pprint(config.as_dict())

# load best metrics
best_metrics = load_best_metrics(MODEL_DIR)
# load model
model = load_model(MODEL_DIR, config=config)
model.test_threshold = best_metrics['best_validation_threshold']

# create dataset reader
dataset_reader = load_dataset_reader(config=config)

# create iterator
# This will take time for the first time as it will create the cache
iterator = load_iterator(config=config)
generator = create_onepass_generator(iterator, dataset_reader)

from typing import Iterator, Union, Any
from allennlp.nn import util as nn_util

def predict_loop_or_load(
        model: Model,
        dataset_iterator: Iterator,
        device: str = 'cpu',
        output_file: Union[str, Path] = 'output.jsonl',
        force_repredict: bool = False) -> List[Dict[str, Any]]:
    """
    Checks if results are already present in the output file. If the file exists reads it and returns
    the contents. If it does not, runs the prediction loop and populates the file and returns results.
    """
    # check
    output_file: Path = Path(output_file)  # type: ignore

    if output_file.exists():
        if output_file.is_file():
            logger.info("{} file already exists...")

            if force_repredict:
                logger.info(
                    "force_repredict is True. Hence repredicting and overwritting"
                )
            else:
                logger.info("Reading results from the existing file ")

                return load_outputs(output_file)

    # Predict
    device = 'cpu'

    if device is 'cpu':
        device_instance = torch.device('cpu')
        device_int = -1
    else:
        device_instance = torch.device('cuda', 0)
        device_int = 0
    model = model.to(device=device_instance)
    model.eval()
    model.test()
    with open(output_file, 'w') as f:
        logger.info("Starting predictions ...")

        for i, input_batch in enumerate(tqdm.tqdm(dataset_iterator)):
            input_batch_on_device = nn_util.move_to_device(
                input_batch, device_int)
            result = model.forward(**input_batch_on_device)
            print(model.get_metrics())
        logger.info('Prediction complete')
        metrics = model.get_metrics()
        logger.info("Test metrics\n")
        print(metrics)
        json.dump(metrics, f)
        logger.info(f"Writing metrics to {output_file}")
    return metrics

results = predict_loop_or_load(model, generator, device='cpu', output_file=MODEL_DIR+'test_outputs.jsonl', force_repredict=True)

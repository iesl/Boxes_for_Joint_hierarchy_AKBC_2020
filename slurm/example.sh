#!/bin/bash
#SBATCH --job-name=kb_completion-%j
#SBATCH --gres=gpu:1
#SBATCH --mem=16000
#SBATCH --output=/mnt/nfs/work1/mccallum/dhruveshpate/kb_completion/models/logs/sigmoid_boxes_single_negative_sample/%j.out

module load cuda90/toolkit 

cd /mnt/nfs/work1/mccallum/dhruveshpate/kb_completion/models

source /mnt/nfs/work1/mccallum/dhruveshpate/kb_completion/models/.venv_models/bin/activate


cd /mnt/nfs/work1/mccallum/dhruveshpate/kb_completion/models
export PYTHONPATH=PYTHONPATH:/mnt/nfs/work1/mccallum/dhruveshpate/kb_completion/models

exec allennlp train configs/gypsum_default_training_with_rank_validation.json -s logs/sigmoid_boxes_single_negative_sample/$1 --include-package models --include-package datasets --overrides "{\"trainer.num_epochs\":20, \"model.margin\":3.0, \"trainer.cuda_device\":0}"

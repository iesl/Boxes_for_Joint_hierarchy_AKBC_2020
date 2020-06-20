#!/bin/sh
module load cuda90/toolkit 

cd /mnt/nfs/work1/mccallum/dhruveshpate/kb_completion/models

source /mnt/nfs/work1/mccallum/dhruveshpate/kb_completion/models/.venv_models/bin/activate

cd /mnt/nfs/work1/mccallum/dhruveshpate/kb_completion/models
export PYTHONPATH=PYTHONPATH:/mnt/nfs/work1/mccallum/dhruveshpate/kb_completion/models

export WANDB_TAGS=head_to_tail,conditional,max_margin,SigmoidBoxTensor,val_avg_rank,WN18RR


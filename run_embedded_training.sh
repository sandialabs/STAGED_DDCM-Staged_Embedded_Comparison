#!/bin/bash

python embedded_training_parallel.py \
  --output-dir data/study_replicates \
  --num-replicates 50 \
  --max-workers 10 \
  --num-starts 10 \
  --val-every 100 \
  --patience 4
#  --overwrite

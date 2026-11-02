#!/bin/bash

python staged_generalization_parallel.py \
    --study-dir data/study_replicates \
    --num-replicates 50 \
    --max-workers 8 \
   --overwrite

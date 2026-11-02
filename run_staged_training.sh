#!/bin/bash

n_reps=50

python da_training_parallel.py \
  --study-dir data/study_replicates \
  --num-replicates $n_reps \
  --max-workers 10 \
  --save-enkf-ensemble \
  --estimate-process-noise \
# --overwrite
#  --save-enkf-sd \

python staged_training_parallel.py \
--study-dir data/study_replicates \
--num-replicates $n_reps \
--num-starts 10 \
--max-workers 10 \
--base-staged-seed 50000 \
--val-every 100 \
--patience 4
#--overwrite
#--diagnostic-mode \

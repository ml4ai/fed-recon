#!/bin/bash

results_dir=/tmp/icarl-fed-recon-results

python mtm/fed_recon/eval_model.py \
    --config configs/icarl-eval-fed-recon.json \
    --output_path ${results_dir}

#!/bin/bash

results_dir=/tmp/icarl-fed-recon-results

python fed_recon/benchmark/eval_model.py \
    --config configs/icarl-eval-fed-recon.json \
    --output_path ${results_dir}

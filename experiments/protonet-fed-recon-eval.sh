#!/bin/bash

results_dir=/tmp/protonet-fed-recon-results

python mtm/fed_recon/eval_model.py \
    --config configs/protonet-eval-fed-recon.json \
    --output_path ${results_dir}

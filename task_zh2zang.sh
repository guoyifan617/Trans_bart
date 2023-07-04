#!/bin/bash
set -e -o pipefail -x
source /home/yfguo/scripts/init_conda.sh
conda activate Trans_base

python train_bart_zh2zang.py
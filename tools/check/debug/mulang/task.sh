#!/bin/bash

set -e -o pipefail -x

export gpuid=$1
export task=$2

export $srcd

cd wspool/v$gpuid/
#python tools/extract_data.py $srcd $task cache/opus/$task
bash scripts/mktrain.sh $2
cp cache/opus/$task/fbind.py .
sed -i "s/data_id = \"w14ed32\"/data_id = \"opus\/$task\"/g" cnfg/base.py
python train.py
python tools/average_model.py expm/opus/$task/std/base/avg.h5 expm/opus/$task/std/base/checkpoint_0.h5 expm/opus/$task/std/base/checkpoint_1.h5 expm/opus/$task/std/base/checkpoint_2.h5 expm/opus/$task/std/base/checkpoint_3.h5 expm/opus/$task/std/base/last.h5
rm -fr expm/opus/$task/std/base/checkpoint_* expm/opus/$task/std/base/last.h5 expm/opus/$task/std/base/init.h5 expm/opus/$task/std/base/train_*.h5
sed -i "s/data_id = \"opus\/$task\"/data_id = \"w14ed32\"/g" cnfg/base.py

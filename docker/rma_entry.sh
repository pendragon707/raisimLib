#!/bin/bash

source /root/miniconda3/etc/profile.d/conda.sh
conda activate rma

cd /workspace/rma/raisimGymTorch
pip install -e .

python setup.py develop
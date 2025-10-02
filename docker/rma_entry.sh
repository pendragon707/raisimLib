#!/bin/bash

source /root/miniconda3/bin/activate
conda activate rma

pip install cbor2 mujoco crcmod
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

cd /workspace
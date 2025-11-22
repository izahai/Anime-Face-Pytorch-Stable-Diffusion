#!/bin/bash

export PYTHONPATH="$(pwd):$PYTHONPATH"

python tools/train_vae.py --config configs/vae.yaml
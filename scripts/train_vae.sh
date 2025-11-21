#!/bin/bash

export PYTHONPATH="$(pwd):$PYTHONPATH"

python train_vae.py --config configs/vae_config.json
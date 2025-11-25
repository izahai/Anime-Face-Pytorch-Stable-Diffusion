#!/bin/bash

export PYTHONPATH="$(pwd):$PYTHONPATH"

python tools/train_ldm.py --config configs/ldm.yaml
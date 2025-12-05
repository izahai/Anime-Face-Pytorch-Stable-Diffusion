#!/bin/bash

export PYTHONPATH="$(pwd):$PYTHONPATH"

CKPT="vae_logs/checkpoints/epoch49.pt"
OUT="vae_logs/samples/epoch49.png"

#For generating a grid
python tools/run_vae.py \
    --ckpt $CKPT \
    --out $OUT \
    --grid \
    --n 100 \
    --nrow 10


echo "Sampling complete: saved to $OUT"
#!/bin/bash

export PYTHONPATH="$(pwd):$PYTHONPATH"

CKPT="vae_logs/checkpoints/epoch36.pt"
OUT="vae_logs/samples/grid21.png"

# For generating a grid
python tools/run_vae.py \
    --ckpt $CKPT \
    --out $OUT \
    --grid \
    --n 10 \
    --nrow 5

echo "Sampling complete: saved to $OUT"
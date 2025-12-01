#!/bin/bash

export PYTHONPATH="$(pwd):$PYTHONPATH"

CKPT="vae_logs/checkpoints/best_vae_epoch_56_v2.pt"
OUT="vae_logs/samples/recon.png"

# For generating a grid
# python tools/run_vae.py \
#     --ckpt $CKPT \
#     --out $OUT \
#     --grid \
#     --n 10 \
#     --nrow 5

python tools/run_vae.py \
    --ckpt $CKPT \
    --recon \
    --input_folder data/images \
    --recon_n 20 \
    --nrow 5 \
    --out $OUT


echo "Sampling complete: saved to $OUT"
#!/bin/bash

export PYTHONPATH="$(pwd):$PYTHONPATH"

CKPT="vae_logs/checkpoints/genshin_96.pt"
OUT="vae_logs/samples/vae_genshin.png"

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
    --input_folder genshin/images \
    --recon_n 70 \
    --nrow 10 \
    --out $OUT


echo "Sampling complete: saved to $OUT"
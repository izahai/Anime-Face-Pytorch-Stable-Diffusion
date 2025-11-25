#!/bin/bash

export PYTHONPATH="$(pwd):$PYTHONPATH"

VAE_CKPT="vae_logs/checkpoints/epoch49.pt"
UNET_CKPT="vae_logs/checkpoints/best_ldm_epoch_8_b32.pt"
OUT="vae_logs/samples/grid_ldm_8b32.png"

# For generating a grid
python tools/run_ldm.py \
  --vae_ckpt $VAE_CKPT \
  --unet_ckpt $UNET_CKPT \
  --out $OUT \
  --num_samples 4 \
  --nrow 4 \
  --image_size 64 \
  --z_ch 4 \
  --base_ch 64

echo "Sampling complete: saved to $OUT"
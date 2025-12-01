#!/bin/bash

export PYTHONPATH="$(pwd):$PYTHONPATH"

VAE_CKPT="vae_logs/checkpoints/best_vae_epoch_56_v2.pt"
UNET_CKPT="vae_logs/checkpoints/ldm_epoch_49.pt"
OUT="vae_logs/samples/ldm_epoch_49.png"

# For generating a grid
python tools/run_ldm.py \
  --vae_ckpt $VAE_CKPT \
  --unet_ckpt $UNET_CKPT \
  --out $OUT \
  --num_samples 25 \
  --nrow 5 \
  --image_size 64 \
  --z_ch 4 \
  --base_ch 64

echo "Sampling complete: saved to $OUT"
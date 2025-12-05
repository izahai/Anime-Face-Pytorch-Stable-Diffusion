#!/bin/bash

export PYTHONPATH="$(pwd):$PYTHONPATH"

CKPT="vae_logs/checkpoints/vae_gan_128ch_8z_epoch24.pt"
OUT="vae_logs/samples/vae_gan_128ch_8z_epoch24_recon.png"

#For generating a grid
# python tools/run_vae.py \
#     --ckpt $CKPT \
#     --out $OUT \
#     --grid \
#     --n 100 \
#     --nrow 10

python tools/run_vae.py \
    --ckpt $CKPT \
    --out $OUT \
    --input_folder "data/images"\
    --recon \
    --n 300 \
    --nrow 15 \
    --z_ch 8 \
    --base_ch 128 \
    --num_head 8 \

echo "Sampling complete: saved to $OUT"
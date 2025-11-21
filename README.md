# Anime-Face-Pytorch-Stable-Diffusion
This project builds a mini Stable Diffusion pipeline with pytorch, starting with training a Variational Autoencoder (VAE) on a 64x64 anime face dataset.

This repository is designed for **learning**, **research**, and **fun**.

## Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
The code supports *automatic* dataset downloading (both locally and in Google Colab) using **KaggleHub**.

If automatic download fails:

Download the dataset here:
[https://www.kaggle.com/datasets/splcher/animefacedataset](https://www.kaggle.com/datasets/splcher/animefacedataset)

Then place the images inside:

```
data/images/
```

### 3. Train the VAE
Run training:

```bash
bash scripts/train_vae.sh
```

All outputs are saved to:

```
outputs/vae/
├── samples/        # reconstruction preview images
└── checkpoints/    # epoch + best model weights
```

### 4. Inference VAE
Run:

```bash
bash scripts/infer_vae.sh
```

This generates:

* Random anime faces sampled from the latent space
* A grid of generated samples

## Architecture
### 1. Variational Autoencoder (VAE)
The VAE learns to compress and reconstruct 64×64 anime faces:

```
Image (64×64)
   ↓
Encoder
   ↓
Latent z (4×8×8)
   ↓
Decoder
   ↓
Reconstructed Image
```

This VAE will be used later as the latent space encoder/decoder for your **mini Stable Diffusion**.

### 2. Latent Diffusion
More updates coming soon…

## Paper Reference
**Auto-Encoding Variational Bayes**
Kingma & Welling, 2013

## License

This project is licensed under the **MIT License**.
See the `LICENSE` file for details.
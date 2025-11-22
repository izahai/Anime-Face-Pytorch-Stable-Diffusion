# Anime-Face-Pytorch-Stable-Diffusion
This project builds a **mini Stable Diffusion pipeline using Pytorch**, starting with training a Variational Autoencoder (VAE) on a 64x64 anime face dataset.

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
#### Train on Google Colab (on T4 GPU)
Open and run the notebook:

```
notebook/train_vae.ipynb
````

#### Train locally

```bash
bash scripts/train_vae.sh
````

All outputs are saved to:

```
outputs/vae/
├── samples/        # reconstruction preview images
└── checkpoints/    # epoch + best model weights
```

You can modify the training hyperparameters in the file `configs/vae_config.json`.

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
Image (3x64×64)
   ↓
Encoder (Downsample x4)
   ↓
Latent z (4×16×16)
   ↓
Decoder (Upsample x4)
   ↓
Reconstructed Image
```

This VAE will be used later as the latent space encoder/decoder for your **mini Stable Diffusion**.

### 2. Latent Diffusion
More updates coming soon…

## Paper Reference
**Auto-Encoding Variational Bayes** Kingma & Welling, 2013

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
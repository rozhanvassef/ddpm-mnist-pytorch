# ddpm-mnist-pytorch
PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) for MNIST digit generation

# DDPM Implementation on MNIST Dataset

A PyTorch implementation of Denoising Diffusion Probabilistic Models (DDPM) trained on the MNIST dataset, based on the [labml.ai DDPM tutorial](https://nn.labml.ai/diffusion/ddpm/index.html).

## Overview

This project implements the DDPM algorithm as described in the paper ["Denoising Diffusion Probabilistic Models"](https://arxiv.org/abs/2006.11239) by Ho et al. The model learns to generate MNIST digit images through a diffusion process that gradually adds noise to images during training, then learns to reverse this process for generation.

## Demo

### Denoising Process Video
![Denoising Animation](path/to/denoising_video.gif)

*Watch the model transform random noise into a recognizable digit through 1000 denoising steps.*

## Features

- Complete DDPM implementation with U-Net architecture
- Training pipeline for MNIST dataset
- Image generation and sampling utilities
- Step-by-step visualization of the denoising process
- Configurable model architecture and hyperparameters
- Checkpoint saving and model loading

## Model Architecture

The implementation uses a U-Net architecture with:
- **Residual blocks** with group normalization
- **Self-attention mechanisms** at multiple resolutions
- **Time step embeddings** using sinusoidal encoding
- **Skip connections** between encoder and decoder paths

### Training

The model is trained on the MNIST dataset with the following default hyperparameters:
```python
image_size: 32x32
diffusion_steps: 1000
batch_size: 64
learning_rate: 2e-5
epochs: 3 (configurable)
```

Run the training notebook to train the model. Training for 3 epochs takes approximately 20-30 minutes on a GPU.

### Generation

After training, use the Sampler class to generate new MNIST-style digits:
```python
# Load trained model
sampler = Sampler(diffusion, image_channels=1, image_size=32, device=device)

# Generate samples
sampler.sample_animation(n_frames=100, create_video=True)
```

Each generation run produces unique outputs as the process starts from random noise.

## Training Configuration
```python
epochs: 3                      # Number of training epochs
batch_size: 64                 # Training batch size
n_steps: 1000                  # Diffusion timesteps
learning_rate: 2e-5           # Adam optimizer learning rate
n_channels: 64                 # Base channels in U-Net
channel_multipliers: [1,2,2,4] # Channel scaling at each resolution
```

## Key Implementation Details

### Forward Diffusion Process
The model gradually adds Gaussian noise to images over T timesteps:
```
q(x_t|x_0) = N(x_t; √(ᾱ_t)x_0, (1-ᾱ_t)I)
```

### Reverse Denoising Process
The neural network learns to predict and remove noise:
```
p_θ(x_{t-1}|x_t) = N(x_{t-1}; μ_θ(x_t,t), σ_t²I)
```

### Training Objective
Simplified loss function (MSE between true and predicted noise):
```
L_simple(θ) = E_{t,x_0,ε}[||ε - ε_θ(√(ᾱ_t)x_0 + √(1-ᾱ_t)ε, t)||²]
```

## Technical Specifications

- **Framework:** PyTorch
- **Model Parameters:** ~35M parameters
- **Training Time:** ~7 minutes per epoch (on V100 GPU)
- **Inference Time:** ~50 seconds per sample (1000 steps)
- **Memory Requirements:** ~2GB GPU VRAM

## Limitations

- Limited training (3 epochs) may result in:
  - Occasional low-quality generations
  - Bias toward certain digits
  - Less diverse outputs
- Sequential denoising process is computationally intensive
- Generated images are 32×32 resolution

## Future Improvements

- [ ] Increase training epochs for better quality
- [ ] Implement DDIM for faster sampling
- [ ] Add conditional generation (class-conditional)
- [ ] Experiment with different noise schedules
- [ ] Scale to higher resolution images
- [ ] Implement latent diffusion for efficiency

## References

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - Ho et al., 2020
- [labml.ai DDPM Implementation](https://nn.labml.ai/diffusion/ddpm/index.html)
- [Understanding Diffusion Models](https://arxiv.org/abs/2208.11970) - Luo, 2022


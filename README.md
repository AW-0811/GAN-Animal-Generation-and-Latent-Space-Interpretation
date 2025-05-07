# **GAN Animal Generation and Latent Space Interpretation**

A dual-approach GAN project exploring both Conditional (ACGAN) and Unconditional architectures to generate realistic animal faces, while analyzing the effects of class conditioning and latent space structure.

---

## **Table of Contents**
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Code Structure](#code-structure)
  - [Conditional GAN](#conditional-gan)
  - [Vanilla GAN](#vanilla-gan)

---

## **Project Overview**

This project implements both **Conditional GAN (ACGAN)** and **Vanilla (Unconditional) GAN** architectures for generating realistic animal face images using the [AFHQv2 512×512 dataset](https://www.kaggle.com/datasets/dimensi0n/afhq-512).  
- The **Conditional GAN** allows class-controlled generation.
- The **Vanilla GAN** serves as a baseline without label guidance.

Together, these models provide insight into how class conditioning impacts image quality, training stability, and interpretability of the latent space.

---

## **Installation**

To run this project locally:

```bash
# 1. Clone the repository
git clone https://github.com/AW-0811/GAN-Animal-Generation-and-Latent-Space-Interpretation.git
cd GAN-Animal-Generation-and-Latent-Space-Interpretation

# 2. Create a virtual environment
python -m venv gan_env
source gan_env/bin/activate  # On Windows use: gan_env\Scripts\activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
pip install -r requirements.txt
```

> Ensure you have GPU support (CUDA) with your PyTorch installation for best performance. I had access to A-100 GPUs through Northwestern QUEST.

---

## **Usage Guide**

This guide outlines the correct execution flow for both GAN models.  
**Note:** You may need to modify dataset and weight file paths depending on your environment.

### **Step 1: Train the GAN**

- **Conditional GAN**
  ```bash
  python ACGAN.py
  ```
- **Vanilla GAN**
  ```bash
  python "Unconditional GAN.py"
  ```

> Outputs: `.pth` model weights, image snapshots across epochs, loss plots.

---

### **Step 2: Generate Images**

- **Conditional GAN**
  ```bash
  python Generateimages.py
  ```
- **Vanilla GAN**
  ```bash
  python "save images and vectors.py"
  ```

> Outputs: Final generated images and corresponding latent vectors (`.pt` files).

---

### **Step 3: Interpolate Latent Space**

- **Conditional GAN**
  ```bash
  python interpolate.py
  ```
- **Vanilla GAN**
  ```bash
  python "latent space exploration.py"
  ```

> Outputs: Smooth image transitions across the latent space saved to results folders.

---

## **Code Structure**

### **Conditional GAN**
- `ACGAN.py` — Model training script 
- `Generateimages.py` — Image generation script using saved weights  
- `interpolate.py` — Latent vector interpolation script
- `Images over epochs/` — Checkpointed images from training runs I conducted 
- `Interpolation results/` — Some of the better interpolation samples from my testing
- `Plots/` — Loss curves and comparison plots for the model I trained
- `Generated images and vectors/` — Some Image–vector pairs I generated with the epoch 80 generator  
- Saved models: `epoch_80.pth` and `epoch_99.pth` (I personally found better generalization at epoch 80)

---

### **Vanilla GAN**
- `Unconditional GAN.py` — Model training script
- `save images and vectors.py` — Image generation script with vector saving  
- `latent space exploration.py` — Script for Interpolation across generated vectors  
- `Fake images over epochs/` — Checkpointed images from my training run 
- `Interpolation examples/` — Latent traversal outputs for some vectors that I personally found looked the best
- `Generated images/` — Generated images from the generation script
- `generated vectors/` — Stored latent vectors from the generation script
- Saved models: `epoch_99.pth`

---

**Need help setting up or got any queries related to this project?** Feel free to reach out to me via GitHub issues or discussions.




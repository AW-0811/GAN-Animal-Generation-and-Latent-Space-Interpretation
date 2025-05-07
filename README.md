## **Table of Contents**
- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage Guide](#usage-guide)
- [Code Structure](#code-structure)
  - [Conditional GAN](#conditional-gan)
  - [Vanilla GAN](#vanilla-gan)

---

## **Project Overview**
This project explores both Conditional (ACGAN) and Unconditional GAN architectures to generate realistic animal images using the [AFHQv2 512×512 animal faces dataset](https://www.kaggle.com/datasets/dimensi0n/afhq-512). The conditional model allows for class-controlled image generation, while the unconditional counterpart serves as a baseline of sorts, to evaluate the performance of a GAN without class guidance. Together, they provide a holistic view of how class conditioning and latent space structure affect output quality and interpretability of the latent space.

---

## **Code Structure**

### **Conditional GAN**
1. `ACGAN.py` — for training the model  
2. `Generateimages.py` — generates images by loading the trained generator through the `.pth` file  
3. `interpolate.py` — for latent space exploration  
4. `Images over epochs` — folder showcasing generated images throughout checkpoints during training  
5. `Interpolation results` — folder containing some of the better interpolation results  
6. `Plots` — folder for loss graphs and image comparisons  
7. `Generated images and vectors` — folder containing generated images along with their corresponding latent vectors  
8. Saved model weights for the discriminator and generator at epoch 80 and 99 (generalization at epoch 80 found to be better)

---

### **Vanilla GAN**
1. `Unconditional GAN.py` — for training the model  
2. `save images and vectors.py` — generates images by loading the trained generator through the `.pth` file  
3. `latent space exploration.py` — for latent space exploration  
4. `Fake images over epochs` — folder showcasing generated images throughout checkpoints during training  
5. `Interpolation examples` — folder containing the interpolation results  
6. `Generated images` — folder containing generated images  
7. `generated vectors` — folder containing the generated vectors  
8. Saved model weights for the discriminator and generator at epoch 99

---

## **Installation**

Follow the steps below to set up the project on your local machine:

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
# **Usage Guide**

This guide outlines the correct order of executing scripts in the **GAN Animal Generation and Latent Space Interpretation** project for both Conditional and Unconditional GANs. **PLEASE NOTE** that file paths will have to be modified to the location of your dataset/model weights.

---

## **Step-by-step Execution Flow**

### **1. Train the GAN Model**
Start by training the model to generate checkpoints and learn image distributions.

- **Conditional GAN:**
  ```bash
  python ACGAN.py
  ```

- **Vanilla GAN:**
  ```bash
  python "Unconditional GAN.py"
  ```

> Outputs: Model weights (`.pth`), image samples across epochs, and training loss plots.

---

### **2. Generate Images**
Once training is complete, generate synthetic images and optionally save latent vectors for future interpolation.

- **Conditional GAN:**
  ```bash
  python Generateimages.py
  ```

- **Vanilla GAN:**
  ```bash
  python "save images and vectors.py"
  ```

> Outputs: Final image samples and `.pt` files for latent vectors.

---

### **3. Perform Latent Space Interpolation**
Explore how the GAN transitions between different points in the latent space.

- **Conditional GAN:**
  ```bash
  python interpolate.py
  ```

- **Vanilla GAN:**
  ```bash
  python "latent space exploration.py"
  ```

> Outputs: Interpolation images saved to the respective results folders.

---

**Tip:** Ensure the required `.pth` model weights and `.pt` latent vectors exist before running generation or interpolation scripts.




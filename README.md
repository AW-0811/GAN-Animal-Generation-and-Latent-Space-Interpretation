**# GAN Animal Generation and Latent Space Interpretation**

**## Project Overview**
### This project explores both Conditional (ACGAN) and Unconditional GAN architectures to generate realistic animal images using the [AFHQv2 512×512 animal faces dataset](https://www.kaggle.com/datasets/dimensi0n/afhq-512). The conditional model allows for class-controlled image generation, while the unconditional counterpart serves as a baseline of sorts, to evaluate the performance of a GAN without class guidance. Together, they provide a holistic view of how class conditioning and latent space structure affect output quality and interpretability of the latent space.

**## Code Structure**
### The project has two separate folders, namely 'Vanilla GAN' and 'Conditional GAN'. The description for each folder's contents are given below:
#### Conditional GAN:
#### ACGAN.py for training the model.
#### Generateimages.py for generating images by loading the a trained generator through the .pth file.
#### interpolate.py for latent space exploration.
#### 'Images over epochs' folder showcasing generated images throughout checkpoints during training.
#### 'Interpolation results' folder containing some of the better interpolation results I generated.
#### 'Plots' folder for loss graphs and image comparisons
#### 'Generated images and vectors' folder containing generated images along with their corresponding latent vectors










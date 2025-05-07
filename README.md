# **GAN Animal Generation and Latent Space Interpretation**

## **Project Overview**
### This project explores both Conditional (ACGAN) and Unconditional GAN architectures to generate realistic animal images using the [AFHQv2 512×512 animal faces dataset](https://www.kaggle.com/datasets/dimensi0n/afhq-512). The conditional model allows for class-controlled image generation, while the unconditional counterpart serves as a baseline of sorts, to evaluate the performance of a GAN without class guidance. Together, they provide a holistic view of how class conditioning and latent space structure affect output quality and interpretability of the latent space.

## **Code Structure**
### The project has two separate folders, namely 'Vanilla GAN' and 'Conditional GAN'. The description for each folder's contents are given below:
#### Conditional GAN:
1. 'ACGAN.py' for training the model\
2. 'Generateimages.py' for generating images by loading the trained generator through the .pth file.\
3. 'interpolate.py' for latent space exploration.\
4. 'Images over epochs' folder showcasing generated images throughout checkpoints during training.\
5. 'Interpolation results' folder containing some of the better interpolation results I generated.\
6. 'Plots' folder for loss graphs and image comparisons.\
7. 'Generated images and vectors' folder containing generated images along with their corresponding latent vectors.\
8. Saved model weights for the discriminator and generator at epoch 80 and 99 (I found the generalization at epoch 80 better than epoch 99)


#### Vanilla GAN:
1. 'Unconditional GAN.py' for training the model\
2. 'save images and vectors.py' for generating images by loading the trained generator through the .pth file.\
3. 'latent space exploration.py' for latent space exploration.\
4. 'Fake images over epochs' folder showcasing generated images throughout checkpoints during training.\
5. 'Interpolation examples' folder containing the interpolation results.\
6. 'Generated images' folder containing generated images.\
7. 'generated vectors' folder containing the generated vectors.\
8. Saved model weights for the discriminator and generator at epoch 99.












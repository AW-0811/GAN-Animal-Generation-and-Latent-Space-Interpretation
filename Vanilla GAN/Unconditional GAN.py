import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
from torch.utils.data import DataLoader

# Hyperparameters
image_size = 64 # Image size
batch_size = 256 # Batch size
nz = 256  # Size of z i.e. latent vector
ngf = 128  # Size of feature maps in generator
df = 128   # Size of feature maps in discriminator
nc = 3    # No. of channels (RGB in this case)
epochs = 100 # No. of epochs
lr = 0.0002 # Learning rate
beta1 = 0.5 # Beta value for Adam

# GPU Config
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Output directory(change path if needed)
os.makedirs("Unconditional_GAN_outputs", exist_ok=True)

# Data preprocessing followed by loading
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.ImageFolder(root="/home/zyv2949/animals", transform=transform) # CHANGE PATH TO DATASET 
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

# Generator architecture
# Input size of 256,1,1
# Final output size of 3,64,64
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Discriminator architecture
# Input size of 3,64,64
# Final output is a flattened scalar containing the probability of each image being real
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, df, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(df, df * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(df * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(df * 2, df * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(df * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(df * 4, df * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(df * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(df * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)
    
# Initializing G and D
generator = Generator().to(device)
discriminator = Discriminator().to(device)

# Losses and optimizers (BCE loss and Adam)
criterion = nn.BCELoss()
optimizerD = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))

# Fixed noise for initial input to generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

# Main training loop
for epoch in range(epochs):
    for i, (real_images, _) in enumerate(dataloader):
        # Train Discriminator
        discriminator.zero_grad()
        real_images = real_images.to(device)
        b_size = real_images.size(0)
        # Creating the labels and backprop
        labels_real = torch.full((b_size,), 1., device=device)
        labels_fake = torch.full((b_size,), 0., device=device)
        output_real = discriminator(real_images).view(-1)
        loss_real = criterion(output_real, labels_real)
        loss_real.backward()
        # generating fake image and computing its loss
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        fake_images = generator(noise)
        output_fake = discriminator(fake_images.detach()).view(-1)
        loss_fake = criterion(output_fake, labels_fake)
        loss_fake.backward()
        #Updating discriminator with losses
        optimizerD.step()
        # Train Generator and backprop after evaluation from Discriminator
        generator.zero_grad()
        output = discriminator(fake_images).view(-1)
        lossG = criterion(output, labels_real)  
        lossG.backward()
        optimizerG.step()
        #printing losses every 50 batches 
        if i % 50 == 0:
            print(f"Epoch [{epoch}/{epochs}] Batch {i}/{len(dataloader)}\tLoss D: {loss_real + loss_fake:.4f}, Loss G: {lossG:.4f}")

    # Saving sample images per epoch
    with torch.no_grad():
        fake = generator(fixed_noise).detach().cpu()
        img_grid = make_grid(fake, padding=2, normalize=True)
        save_image(img_grid, f"/home/zyv2949/output_moredatagan/fake_samples_epoch_{epoch:03d}.png")

    # Saving models per epoch
    torch.save(generator.state_dict(), f"/home/zyv2949/output_moredatagan/generator_epoch_{epoch:03d}.pth")
    torch.save(discriminator.state_dict(), f"/home/zyv2949/output_moredatagan/discriminator_epoch_{epoch:03d}.pth")

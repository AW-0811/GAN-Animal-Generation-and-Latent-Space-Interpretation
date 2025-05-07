import torch
from torchvision.utils import save_image, make_grid
import os

# Generator architecture, same as the training architecture
class Generator(torch.nn.Module):
    def __init__(self, nz=256, ngf=128, nc=3):
        super(Generator, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(ngf * 8),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf * 4),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf * 2),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            torch.nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# You can change number of samples generated here
nz = 256
num_samples = 500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output directories for generated images and their latent vectors
os.makedirs("generated_samples_from_trained_model", exist_ok=True)
os.makedirs("latent_vectors_from_trained_model", exist_ok=True)

# Loading the generator from saved model
generator = Generator(nz=nz).to(device)
generator.load_state_dict(torch.load("generator_epoch_099.pth", map_location=device)) #CHANGE THE PATH TO THE MODEL YOU WANT TO USE
generator.eval()

# Generating and saving 100 images along with their vectors
for i in range(num_samples):
    z = torch.randn(1, nz, 1, 1, device=device)
    
    with torch.no_grad():
        img = generator(z).cpu()

    save_image(img, f"generated_samples_from_trained_model/sample_{i:02d}.png", normalize=True)
    torch.save(z, f"latent_vectors_from_trained_model/z_{i:02d}.pt")

    print(f"Saved sample_{i:02d}.png and z_{i:02d}.pt")

print("Image and latent vector generation complete !")
import torch
from torchvision.utils import save_image, make_grid
import os

# Generator architecture, same as training
class Generator(torch.nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
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

# Change steps
nz = 100
n_steps = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("interpolation_output", exist_ok=True)

# Load generator from trained weights
generator = Generator(nz=nz).to(device)
generator.load_state_dict(torch.load("generator_epoch_099.pth", map_location=device))
generator.eval()

# Load chosen latent vectors (CHANGE PATHS ACCORDINGLY)
z1 = torch.load("latent_vectors/z_61.pt").to(device)
z2 = torch.load("latent_vectors/z_40.pt").to(device)

# Interpolate for different values of alpha within step size range
interpolated = [(1 - alpha) * z1 + alpha * z2 for alpha in torch.linspace(0, 1, steps=n_steps)]
z_interp = torch.cat(interpolated, dim=0)

# Generate and save images
with torch.no_grad():
    images = generator(z_interp).cpu()

grid = make_grid(images, nrow=n_steps, normalize=True, padding=2)
save_image(grid, "interpolation_output/z_61_to_z_40.png") #CHANGE IMAGE NAME IF GENERATING MULTIPLE IMAGES
print("Saved interpolation: 'interpolation_output/z_23_to_z_37.png'")
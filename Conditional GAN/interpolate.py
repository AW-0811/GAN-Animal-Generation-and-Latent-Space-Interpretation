import torch
import torch.nn as nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import os

# Configs
checkpoint_path = 'C:/Users/asimw/OneDrive/Desktop/ADOLACGANOUTPUT/checkpoints/netG_epoch_80.pth' # CHANGE TO PATH OF YOUR DESIRED MODEL
vector1_path = 'best_images/class0_img062.pt'  # VECTOR 1 .pt FILE PATH
vector2_path = 'best_images/class2_img135.pt'  # VECTOR 2 .pt FILE PATH
output_image_path = 'interpolation_result.png'

steps = 8
nz = 100
embedding_dim = 10
ngf = 64
num_classes = 3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator architecture, same as training 
class Generator(nn.Module):
    def __init__(self, nz, num_classes, embedding_dim, ngf):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, embedding_dim)
        input_dim = nz + embedding_dim

        self.main = nn.Sequential(
            nn.ConvTranspose2d(input_dim, ngf * 8, 4, 1, 0, bias=False),
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

            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, label_embedding):
        z = z.view(z.size(0), nz, 1, 1)
        label_embedding = label_embedding.view(label_embedding.size(0), embedding_dim, 1, 1)
        input = torch.cat([z, label_embedding], dim=1)
        return self.main(input)

# Loading generator
netG = Generator(nz, num_classes, embedding_dim, ngf).to(device)
netG.load_state_dict(torch.load(checkpoint_path, map_location=device))
netG.eval()

# Loading latent vectors
data1 = torch.load(vector1_path)
data2 = torch.load(vector2_path)

z1 = data1['z'].unsqueeze(0).to(device)
z2 = data2['z'].unsqueeze(0).to(device)
label1 = data1['label'].item()
label2 = data2['label'].item()

print(f"Interpolating between: '{vector1_path}' (class {label1}) and '{vector2_path}' (class {label2})")

# Interpolation of vectors
interpolated_images = []
alphas = torch.linspace(0, 1, steps).to(device)

for alpha in alphas:
    z_interp = (1 - alpha) * z1 + alpha * z2

    e1 = netG.label_embedding(torch.tensor([label1], device=device))
    e2 = netG.label_embedding(torch.tensor([label2], device=device))
    c_interp = (1 - alpha) * e1 + alpha * e2

    with torch.no_grad():
        img = netG(z_interp, c_interp).cpu()
    interpolated_images.append(img.squeeze(0))

# Creating images and saving
grid = vutils.make_grid(interpolated_images, nrow=steps, normalize=True, padding=2)
plt.figure(figsize=(steps * 2, 2))
plt.axis("off")
plt.title(f"Interpolation from class {label1} to {label2}")
plt.imshow(torch.permute(grid, (1, 2, 0)).numpy())
plt.savefig(output_image_path)
plt.show()

print(f"\n Interpolation result saved to: {output_image_path}")
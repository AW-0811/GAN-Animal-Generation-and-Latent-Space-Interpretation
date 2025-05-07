import torch
import torch.nn as nn
import torchvision.utils as vutils
import os

# Configs
checkpoint_path = 'C:/Users/asimw/OneDrive/Desktop/ADOLACGANOUTPUT/checkpoints/netG_epoch_80.pth' # CHANGE PATH TO DESIRED MODEL
output_folder = 'generated_images'
samples_per_class = 50 # Only need to change this to gen more images
nz = 100
num_classes = 3
embedding_dim = 10
ngf = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Generator architecture from training
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

    def forward(self, noise, labels):
        label_embed = self.label_embedding(labels)
        label_embed = label_embed.view(label_embed.size(0), embedding_dim, 1, 1)
        noise = noise.view(noise.size(0), nz, 1, 1)
        input = torch.cat([noise, label_embed], 1)
        return self.main(input)


os.makedirs(output_folder, exist_ok=True)
print(f"Saving generated images and vectors in: {output_folder}")

# Loading generator from saved model
netG = Generator(nz=nz, num_classes=num_classes, embedding_dim=embedding_dim, ngf=ngf).to(device)
netG.load_state_dict(torch.load(checkpoint_path, map_location=device))
netG.eval()

# Loop for generating images for each class, along with the corresponding latent vector
img_counter = 0

for class_idx in range(num_classes):
    print(f"Generating samples for class {class_idx}...")

    noise = torch.randn(samples_per_class, nz, device=device)
    labels = torch.full((samples_per_class,), class_idx, dtype=torch.long, device=device)

    with torch.no_grad():
        fake_images = netG(noise, labels).detach().cpu()

    for i in range(samples_per_class):
        img_tensor = fake_images[i]
        img_name = f'class{class_idx}_img{i:03d}'
        img_path = os.path.join(output_folder, img_name + '.png')
        vec_path = os.path.join(output_folder, img_name + '.pt')
        vutils.save_image(img_tensor, img_path, normalize=True)
        data = {
            'z': noise[i].cpu(),
            'label': labels[i].cpu()
        }
        torch.save(data, vec_path)
        img_counter += 1

print(f" Saved {img_counter} images and vectors to '{output_folder}'")
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import os
import time

dataset_root = '/home/zyv2949/animals' # PLEASE CHANGE TO DATASET DIRECTORY
image_size = 64 # Image size
batch_size = 64 # Batch size
nz = 100 # Size of the latent z vector
num_classes = 3  # No. of classes (cat, dog, wild in this case)
embedding_dim = 10 # No. of Dimensions for class embedding
ngf = 64 # Size of feature maps in generator
ndf = 64  # Size of feature maps in discriminator
num_epochs = 100  # Number of epochs
lr = 0.0002 # Learning rate 
beta1 = 0.5 # Beta hyperparameter for Adam optimizer
workers = 2  # Number of workers for dataloader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Output directory for generated images and model checkpoints (PLEASE CHANGE PATHS ACCORDING TO YOUR SETUP)
output_dir = '/home/zyv2949/ADOLACGANOUTPUT'
os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
os.makedirs(os.path.join(output_dir, 'checkpoints'), exist_ok=True)


# Transformations for preprocessing
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Loading the dataset
dataset = datasets.ImageFolder(root=dataset_root, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=workers, drop_last=True)
print(f"Dataset loaded successfully from {dataset_root}. Found {len(dataset)} images in {len(dataset.classes)} classes: {dataset.classes}")
if len(dataset.classes) != num_classes:
    print(f"Number of classes does not match the number of folders found in dataset")


#Initialising weights using normal distribution, hoping to improve convergence 
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# Generator architecture
# input size of 100 noise + 10 class embedding per batch
#output is 3x64x64 per batch
class Generator(nn.Module):
    def __init__(self, nz, num_classes, embedding_dim, ngf):
        super(Generator, self).__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.nz = nz
        self.ngf = ngf
        self.label_embedding = nn.Embedding(num_classes, embedding_dim)
        input_dim = nz + embedding_dim #Combining latent dimensions with class embedding

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
        c = self.label_embedding(labels)
      
        c = c.view(c.size(0), self.embedding_dim, 1, 1) 
        noise = noise.view(noise.size(0), self.nz, 1, 1) 

        x = torch.cat([noise, c], 1) 
        return self.main(x)

# Discriminator Architecture
# input is 3x64x64 per batch
# output is real/fake value + class label logits
class Discriminator(nn.Module):
    def __init__(self, num_classes, ndf):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.ndf = ndf

        self.main = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.source_output = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self.class_output = nn.Sequential(
             nn.Conv2d(ndf * 8, num_classes, 4, 1, 0, bias=False)

        )


    def forward(self, img):
        features = self.main(img)
        source = self.source_output(features).view(-1, 1).squeeze(1)
        clss = self.class_output(features).view(-1, self.num_classes)
        return source, clss


# Initilizing G and D, and adding weights
netG = Generator(nz, num_classes, embedding_dim, ngf).to(device)
netD = Discriminator(num_classes, ndf).to(device)
netG.apply(weights_init)
netD.apply(weights_init)

# Loss functions (BCE for real/fake, and multi class cross entropy loss for class label)
adversarial_loss = nn.BCELoss() 
auxiliary_loss = nn.CrossEntropyLoss()

#Initial noise for generator
fixed_noise = torch.randn(batch_size, nz, device=device)
# Preparing fixed labels so each class appears in the batch
fixed_labels_indices = np.array([i % num_classes for i in range(batch_size)])
fixed_labels = torch.LongTensor(fixed_labels_indices).to(device)
print(f"Fixed labels for visualization (first 10): {fixed_labels_indices[:10]}")

#Setting real/fake values and optimiser for D and G
real_label_val = 1.
fake_label_val = 0.
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))


# Main training loop
print("Starting Training Loop...")
start_time = time.time()

img_list = []
G_losses = []
D_losses = []
D_acc = []

iters = 0

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    d_loss_epoch = 0.0
    g_loss_epoch = 0.0
    correct_real = 0
    total_real = 0

    for i, data in enumerate(dataloader, 0):
        netD.zero_grad()
        real_cpu = data[0].to(device)
        real_labels = data[1].to(device) 
        b_size = real_cpu.size(0)
        # Using real data to train auxiliary classifier, hoping to improve prediction accuracy
        label_adv = torch.full((b_size,), real_label_val, dtype=torch.float, device=device)
        source_output, class_output = netD(real_cpu)

        errD_real_adv = adversarial_loss(source_output, label_adv)
        errD_real_aux = auxiliary_loss(class_output, real_labels)
        errD_real = errD_real_adv + errD_real_aux 
        errD_real.backward()
        D_x = source_output.mean().item()

        pred_real = torch.argmax(class_output, 1)
        correct_real += (pred_real == real_labels).sum().item()
        total_real += b_size
        #Discriminator training on fake image before G updates
        noise = torch.randn(b_size, nz, device=device)
        fake_labels_indices = np.random.randint(0, num_classes, b_size)
        fake_labels = torch.LongTensor(fake_labels_indices).to(device)

        fake = netG(noise, fake_labels)
        label_adv.fill_(fake_label_val)
        source_output, class_output = netD(fake.detach())
        errD_fake_adv = adversarial_loss(source_output, label_adv)
        errD_fake = errD_fake_adv
        errD_fake.backward()
        D_G_z1 = source_output.mean().item() 
        errD = errD_real + errD_fake
        d_loss_epoch += errD.item()

        # Updating Discriminator
        optimizerD.step()

        netG.zero_grad()
        label_adv.fill_(real_label_val)

        # training gewnerator based on loss from discriminator
        source_output, class_output = netD(fake) 

        errG_adv = adversarial_loss(source_output, label_adv) 
        errG_aux = auxiliary_loss(class_output, fake_labels) 
        errG = errG_adv + errG_aux
        g_loss_epoch += errG.item()
        errG.backward()
        D_G_z2 = source_output.mean().item() 
        optimizerG.step()

        #Logs for every 50th batch
        if i % 50 == 0:
            print(f'[{epoch}/{num_epochs}][{i}/{len(dataloader)}] '
                  f'Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f} '
                  f'D(x): {D_x:.4f} D(G(z)): {D_G_z1:.4f} / {D_G_z2:.4f} '
                  f'D Acc Real: {(correct_real / total_real * 100):.2f}%')

        # Saving the losses here so we can plot them later
        G_losses.append(errG.item())
        D_losses.append(errD.item())


        iters += 1

    # End of epoch metrics
    epoch_time = time.time() - epoch_start_time
    avg_d_loss = d_loss_epoch / len(dataloader)
    avg_g_loss = g_loss_epoch / len(dataloader)
    avg_d_acc = (correct_real / total_real * 100) if total_real > 0 else 0
    D_acc.append(avg_d_acc)
    print(f"Epoch {epoch} finished in {epoch_time:.2f}s. Avg Loss_D: {avg_d_loss:.4f}, Avg Loss_G: {avg_g_loss:.4f}, Avg D Acc Real: {avg_d_acc:.2f}%")

    # Generating and saving images using fixed noise and labels
    with torch.no_grad():
        fake_fixed = netG(fixed_noise, fixed_labels).detach().cpu()
    img_grid = vutils.make_grid(fake_fixed, padding=2, normalize=True, nrow=8) # Adjust nrow as needed
    vutils.save_image(img_grid, f"{output_dir}/images/fake_samples_epoch_{epoch:03d}.png")
    img_list.append(img_grid)

    # Saving both models at checkpoints
    if (epoch % 10 == 0) or (epoch == num_epochs - 1):
        torch.save(netG.state_dict(), f'{output_dir}/checkpoints/netG_epoch_{epoch}.pth')
        torch.save(netD.state_dict(), f'{output_dir}/checkpoints/netD_epoch_{epoch}.pth')

# Post training time
total_time = time.time() - start_time
print(f"Training finished in {total_time // 60:.0f}m {total_time % 60:.0f}s")

# Plots
plt.figure(figsize=(10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(f"{output_dir}/loss_plot.png")

plt.figure(figsize=(10, 5))
plt.title("Discriminator Accuracy on Real Images During Training")
plt.plot(D_acc, label="D Accuracy (Real)")
plt.xlabel("Epochs")
plt.ylabel("Accuracy (%)")
plt.legend()
plt.savefig(f"{output_dir}/accuracy_plot.png")

#Plot of real vs fake
real_batch = next(iter(dataloader))

plt.figure(figsize=(10, 10))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(), (1, 2, 0)))

plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images (Last Epoch)")
plt.imshow(np.transpose(img_list[-1], (1, 2, 0))) 
plt.savefig(f"{output_dir}/real_vs_fake.png")

print(f"Training complete. Generated images and plots saved in '{output_dir}'")

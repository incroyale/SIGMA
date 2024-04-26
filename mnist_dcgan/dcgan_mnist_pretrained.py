import os
import torch
from torchvision import transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import zipfile
import shutil

num_gpu = 1 if torch.cuda.is_available() else 0

# load the models
from dcgan import Discriminator, Generator

D = Discriminator(ngpu=1).eval()
G = Generator(ngpu=1).eval()

# load weights
D.load_state_dict(torch.load('weights/netD_epoch_99.pth'))
G.load_state_dict(torch.load('weights/netG_epoch_99.pth'))
if torch.cuda.is_available():
    D = D.cuda()
    G = G.cuda()

batch_size = int(input("Enter number of fake images to generate: "))
latent_size = 100

# Define the output folder for fake images
output_folder = "fake_images"
zip_file_name = "fake_images.zip"

if os.path.exists(output_folder):
    shutil.rmtree(output_folder)
if os.path.exists(zip_file_name):
    os.remove(zip_file_name)

# Create a new folder to store PNG images
os.makedirs(output_folder, exist_ok=True)

# Generate fake images
fixed_noise = torch.randn(batch_size, latent_size, 1, 1)
if torch.cuda.is_available():
    fixed_noise = fixed_noise.cuda()
fake_images = G(fixed_noise)

# Convert the images to numpy arrays
fake_images_np = fake_images.cpu().detach().numpy()
fake_images_np = fake_images_np.reshape(fake_images_np.shape[0], 28, 28)

# Save each image as a PNG in the folder
for i, image_np in enumerate(fake_images_np):
    plt.imshow(image_np, cmap='gray')
    plt.grid(False)
    plt.axis('off')
    plt.savefig(os.path.join(output_folder, f"fake_image_{i}.png"), bbox_inches='tight', pad_inches=0)
    plt.close()


# Zip the folder
def zip_folder(folder_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                zipf.write(os.path.join(root, file), arcname=file)


zip_folder(output_folder, zip_file_name)

print(f"All fake images saved as PNGs in '{output_folder}'")
print(f"Compressed folder '{zip_file_name}' created.")
shutil.rmtree(output_folder)

import os
import torch
from torchvision import transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import zipfile
import shutil

# Get the current directory
current_directory = os.getcwd()
zip_file_name = None

for file_name in os.listdir(current_directory):
    if file_name.endswith('.zip') and file_name != 'fake_images.zip':
        zip_file_name = file_name
        break

if zip_file_name:
    with zipfile.ZipFile(zip_file_name, 'r') as zip_ref:
        zip_ref.extractall('dataset')
    os.remove(zip_file_name)
    print(f"Successfully unzipped '{zip_file_name}' into the 'dataset' directory.")
else:
    print("No zip file found.")

import shutil

# Define the path to the dataset folder
dataset_folder = 'dataset'

# Check if the dataset folder exists
if os.path.exists(dataset_folder):
    # Delete all contents of the dataset folder
    for root, dirs, files in os.walk(dataset_folder):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            shutil.rmtree(os.path.join(root, dir))
    print("Successfully deleted all contents of the 'dataset' folder.")
else:
    print("The 'dataset' folder does not exist.")

epochs = int(input("Enter number of epochs: "))
command = f"python training.py --dataset folder --dataroot C:\\Users\\Lenovo\\PycharmProjects\\DCGAN\\SIGMA\\external_ouput\\dataset --imageSize 28 --cuda --outf weights --niter {epochs}"
os.system(command)
num_gpu = 1 if torch.cuda.is_available() else 0

# load the models
from training import Discriminator, Generator

D = Discriminator(ngpu=1).eval()
G = Generator(ngpu=1).eval()


# load weights
#D.load_state_dict(torch.load('weights/netD_trial.pth'))
#G.load_state_dict(torch.load('weights/netG_trial.pth'))
D.load_state_dict(torch.load('weights/netD_epoch_last.pth'))
G.load_state_dict(torch.load('weights/netG_epoch_last.pth'))
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
shutil.rmtree(output_folder)

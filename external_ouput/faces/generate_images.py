import torch
import os
import shutil
import zipfile
from torchvision.utils import save_image
from training import DCGAN

def generate_images(num_images):
    model = DCGAN()
    model.load_state_dict(torch.load('gan_celeba_01.pt'))
    model.eval()

    latent_dim = 100
    output_folder = 'generated_images'
    os.makedirs(output_folder, exist_ok=True)
    z = torch.randn(num_images, latent_dim, 1, 1)
    fake_images = model.generator_forward(z).detach().cpu()

    for i in range(num_images):
        save_image(fake_images[i], os.path.join(output_folder, f'image_{i}.png'))

    zip_filename = 'generated_images.zip'
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for root, _, files in os.walk(output_folder):
            for file in files:
                zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), output_folder))

    shutil.rmtree(output_folder)

generate_images(20)

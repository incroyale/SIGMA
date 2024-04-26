import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import multiprocessing
from helper_train import train_gan_v1
from helper_utils import set_deterministic, set_all_seeds
import zipfile
import os


class DCGAN(torch.nn.Module):

    def __init__(self, latent_dim=100,
                 num_feat_maps_gen=64, num_feat_maps_dis=64,
                 color_channels=3):
        super().__init__()

        self.generator = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, num_feat_maps_gen * 8,
                               kernel_size=4, stride=1, padding=0,
                               bias=False),
            nn.BatchNorm2d(num_feat_maps_gen * 8),
            nn.LeakyReLU(inplace=True),
            #
            # size if latent_dim=100: num_feat_maps_gen*8 x 4 x 4
            #
            nn.ConvTranspose2d(num_feat_maps_gen * 8, num_feat_maps_gen * 4,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(num_feat_maps_gen * 4),
            nn.LeakyReLU(inplace=True),
            #
            # size if latent_dim=100: num_feat_maps_gen*4 x 8 x 8
            #
            nn.ConvTranspose2d(num_feat_maps_gen * 4, num_feat_maps_gen * 2,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(num_feat_maps_gen * 2),
            nn.LeakyReLU(inplace=True),
            #
            # size if latent_dim=100: num_feat_maps_gen*2 x 16 x 16
            #
            nn.ConvTranspose2d(num_feat_maps_gen * 2, num_feat_maps_gen,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.BatchNorm2d(num_feat_maps_gen),
            nn.LeakyReLU(inplace=True),
            #
            # size if latent_dim=100: num_feat_maps_gen x 32 x 32
            #
            nn.ConvTranspose2d(num_feat_maps_gen, color_channels,
                               kernel_size=4, stride=2, padding=1,
                               bias=False),
            #
            # size: color_channels x 64 x 64
            #
            nn.Tanh()
        )

        self.discriminator = nn.Sequential(
            #
            # input size color_channels x image_height x image_width
            #
            nn.Conv2d(color_channels, num_feat_maps_dis,
                      kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(inplace=True),
            #
            # size: num_feat_maps_dis x 32 x 32
            #
            nn.Conv2d(num_feat_maps_dis, num_feat_maps_dis * 2,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(num_feat_maps_dis * 2),
            nn.LeakyReLU(inplace=True),
            #
            # size: num_feat_maps_dis*2 x 16 x 16
            #
            nn.Conv2d(num_feat_maps_dis * 2, num_feat_maps_dis * 4,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(num_feat_maps_dis * 4),
            nn.LeakyReLU(inplace=True),
            #
            # size: num_feat_maps_dis*4 x 8 x 8
            #
            nn.Conv2d(num_feat_maps_dis * 4, num_feat_maps_dis * 8,
                      kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(num_feat_maps_dis * 8),
            nn.LeakyReLU(inplace=True),
            #
            # size: num_feat_maps_dis*8 x 4 x 4
            #
            nn.Conv2d(num_feat_maps_dis * 8, 1,
                      kernel_size=4, stride=1, padding=0),

            # size: 1 x 1 x 1
            nn.Flatten(),

        )

    def generator_forward(self, z):
        img = self.generator(z)
        return img

    def discriminator_forward(self, img):
        logits = model.discriminator(img)
        return logits


if __name__ == '__main__':
    multiprocessing.freeze_support()
    ############################# SETTINGS ##################################
    # Device
    CUDA_DEVICE_NUM = 1
    DEVICE = torch.device(f'cuda:{CUDA_DEVICE_NUM}' if torch.cuda.is_available() else 'cpu')

    # Hyperparameters
    RANDOM_SEED = 42
    GENERATOR_LEARNING_RATE = 0.0002
    DISCRIMINATOR_LEARNING_RATE = 0.0002

    NUM_EPOCHS = int(input("Enter number of epochs: "))
    BATCH_SIZE = 128

    IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS = 64, 64, 3

    set_deterministic
    set_all_seeds(RANDOM_SEED)

    zip_filename = 'smol.zip'
    extracted_folder = 'data/'
    os.makedirs(extracted_folder, exist_ok=True)
    zip_filepath = os.path.join(os.getcwd(), zip_filename)
    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
        zip_ref.extractall(extracted_folder)


    def get_dataloaders_celeba(batch_size, num_workers=0,
                               train_transforms=None,
                               test_transforms=None):
        """
        Change path for the image_folder. Directory structure: given_path -> classes -> all images
        Don't directly point to directory with images.
        """
        if train_transforms is None:
            train_transforms = transforms.ToTensor()

        if test_transforms is None:
            test_transforms = transforms.ToTensor()

        # Define the full dataset
        full_dataset = datasets.ImageFolder(root=r'C:\Users\Lenovo\PycharmProjects\DCGAN\SIGMA\external_ouput\faces\data',
                                            transform=train_transforms)

        # Create data loader for the full dataset
        data_loader = DataLoader(dataset=full_dataset,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 shuffle=True)

        return data_loader


    ################################# Dataset Pre Processing #################################
    custom_transforms = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop((160, 160)),
        torchvision.transforms.Resize([IMAGE_HEIGHT, IMAGE_WIDTH]),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    data_loader = get_dataloaders_celeba(
        batch_size=BATCH_SIZE,
        train_transforms=custom_transforms,
        test_transforms=custom_transforms,
        num_workers=4)

    # Check if CUDA is available
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda:0')  # Use the first GPU (index 0)
    else:
        DEVICE = torch.device('cpu')  # If CUDA is not available, use CPU

    set_all_seeds(RANDOM_SEED)

    model = DCGAN()
    model.to(DEVICE)

    optim_gen = torch.optim.Adam(model.generator.parameters(), betas=(0.5, 0.999), lr=GENERATOR_LEARNING_RATE)
    optim_discr = torch.optim.Adam(model.discriminator.parameters(), betas=(0.5, 0.999), lr=DISCRIMINATOR_LEARNING_RATE)
    log_dict = train_gan_v1(num_epochs=NUM_EPOCHS, model=model, optimizer_gen=optim_gen, optimizer_discr=optim_discr,
                            latent_dim=100,
                            device=DEVICE, train_loader=data_loader, logging_interval=100,
                            save_model='gan_celeba_01.pt')

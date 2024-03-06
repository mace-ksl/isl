import numpy as np
import matplotlib.pyplot as plt
import random
import torch

def plot_random_image(data):
    """
        Input:
            - dataset: numpy array of images (return value of data_set.get_output_images_as_array)
        Method:
            - Plots masks of the dataset from data_set.get_output_images_as_array()
    """
    # Extract images from each channel
    number_images = data.shape[0]
    random_number = random.randint(0, number_images - 1)
    channel1_image = data[random_number, 0, :, :]
    channel2_image = data[random_number, 1, :, :]

    # Plot the images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(channel1_image, cmap='gray') 
    plt.title('Channel 1 Image: Topology')

    plt.subplot(1, 2, 2)
    plt.imshow(channel2_image, cmap='gray') 
    plt.title('Channel 2 Image: Semantic')

    plt.show()

def plot_random_image_from_dataset(dataset):
    """
        Input:
            - dataset: TensorDataset with input + mask images
        Method:
            - Plots a random set of images from a dataset: Input images + associated masks
    """
    random_index = torch.randint(len(dataset), size=(1,)).item()

    # Get the random image and mask
    random_image = dataset[random_index][0]
    random_mask = dataset[random_index][1]

    image_np = random_image.cpu().numpy()
    mask_np = random_mask.cpu().numpy().squeeze()

    plt.figure(figsize=(12, 8))

    for i in range(3):
        plt.subplot(3, 2, i + 1)
        plt.imshow(image_np[i], cmap='gray')
        plt.title(f'Channel {i + 1}')

    for i in range(2):
        plt.subplot(3, 2, i + 4) 
        plt.imshow(mask_np[i], cmap='gray')
        if i == 0:
            plt.title(f'Mask topology {i + 1}')
        else:
            plt.title(f'Mask semantic {i + 1}')
    plt.tight_layout() 
    plt.show()

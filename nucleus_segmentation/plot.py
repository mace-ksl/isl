import numpy as np
import matplotlib.pyplot as plt
import random
import torch

def plot_random_set_of_images(data_train_in,data_train_out,number_plots=1):

    for i in range(number_plots):
        number_images = data_train_in.shape[0]
        random_number = random.randint(0, number_images - 1)

        print(data_train_in.shape,data_train_out.shape)
        input = data_train_in[random_number, :, :, :]
        output = data_train_out[random_number, 0, :, :]

        fig, axes = plt.subplots(1, 4, figsize=(10, 5))

        axes[0].imshow(input[0, :, :], cmap='gray')
        axes[0].axis('off')  
        axes[0].set_title('First Brightfield Channel')


        axes[1].imshow(input[1, :, :], cmap='gray')
        axes[1].axis('off')  
        axes[1].set_title('Second Brightfield Channel')

        axes[2].imshow(input[2, :, :], cmap='gray')
        axes[2].axis('off')  
        axes[2].set_title('Third Phase Contrast Channel')

        axes[3].imshow(output[:, :], cmap='gray')
        axes[3].axis('off')  
        axes[3].set_title('CellPose Channel')

        plt.tight_layout()
        plt.show()

def plot_random_dapi_image(data):
    """
        Input:
            - dataset: numpy array of images (return value of data_set.get_output_images_as_array)
        Method:
            - Plots masks of the dataset from data_set.get_output_images_as_array()
    """
    # Extract images from each channel
    number_images = data.shape[0]
    random_number = random.randint(0, number_images - 1)
    dapi_image = data[random_number, 0, :, :]

    # Plot the images
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 1, 1)
    plt.imshow(dapi_image, cmap='gray') 
    plt.title('DAPI images')
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
    mask_np = random_mask.cpu().numpy()

    plt.figure(figsize=(12, 8))

    for i in range(3):
        plt.subplot(3, 2, i + 1)
        plt.imshow(image_np[i], cmap='gray')
        plt.title(f'Channel {i + 1}')

    
    for i in range(1):
        normalized_array = (mask_np[i] - mask_np[i].min()) / (mask_np[i].max() - mask_np[i].min())
        plt.subplot(3, 2, i + 4) 
        plt.imshow(normalized_array, cmap='gray')
        if i == 0:
            plt.title(f'Mask DAPI')

    plt.tight_layout() 
    plt.show()
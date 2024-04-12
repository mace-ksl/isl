import random
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,TensorDataset
import numpy as np
from scipy import ndimage

import os
from PIL import Image

def randomCrop(img, mask, width=256, height=256):
    assert img.shape[1] >= height
    assert img.shape[2] >= width
    assert img.shape[1] == mask.shape[1]
    assert img.shape[2] == mask.shape[2]

    x = random.randint(0, img.shape[2] - width)
    y = random.randint(0, img.shape[1] - height)

    img = img[:, y:y+height, x:x+width]
    mask = mask[:, y:y+height, x:x+width]

    return img, mask

def crop_dataset(dataset, width=256,height=256):
    cropped_data = []
    cropped_masks = []

    # Number of patches to extract per image
    num_patches_per_image = 5
    path = r"E:\isl\data_set\tiff224"

    # number of files
    files_list = os.listdir(path)
    number_of_files = len(files_list)
    if number_of_files < 1:
        count = number_of_files
    else:
        count = int(number_of_files / 4)

    # Perform random cropping for each image
    for i in range(len(dataset)):
        # Apply random crop transform to both the data and the mask
        for j in range(num_patches_per_image):

            #print(dataset[i][0].shape)
            #print(dataset[i][1].shape)
            ok_crop = False
            while(ok_crop == False):
                cropped_data_patch,cropped_mask_patch = randomCrop(dataset[i][0], dataset[i][1],width=width,height=height)
                
                normalized_mask = (cropped_mask_patch - cropped_mask_patch.min()) / (cropped_mask_patch.max() - cropped_mask_patch.min())
                print(normalized_mask.shape)
                # Assuming normalized_mask is a PyTorch tensor of shape [1, 256, 256] and already normalized
                binary_mask = (normalized_mask > 0.5).squeeze()  # Convert to binary mask and remove channel dimension if exists

                # Find connected components
                labeled_mask, num_features = ndimage.label(binary_mask.cpu().numpy())  # Convert tensor to numpy array and label components

                # Find sizes of connected components
                component_sizes = ndimage.sum(binary_mask.cpu().numpy(), labeled_mask, range(num_features + 1))

                # Check if any component is larger than 1000
                ok = np.any(component_sizes >= 6000)

                binary_mask2 = (normalized_mask < 0.01).squeeze()
                labeled_mask, num_features = ndimage.label(binary_mask2.cpu().numpy())  # Convert tensor to numpy array and label components
                component_sizes = ndimage.sum(binary_mask2.cpu().numpy(), labeled_mask, range(num_features + 1))
                ok2 = np.any(component_sizes >= 4000)
                if ok and ok2:
                    normalized_data = (cropped_data_patch - cropped_data_patch.min()) / (cropped_data_patch.max() - cropped_data_patch.min())

                    ok_crop = True
                    
            #print("---------",cropped_data_patch.shape,cropped_mask_patch.shape)
            #print(cropped_data_patch.shape)
            # Append the cropped patches to the lists
            cropped_data.append(normalized_data.squeeze(0))  # Remove the batch dimension
            cropped_masks.append(normalized_mask.squeeze(0))  # Remove the batch dimension

            
    # Convert the lists to tensors
    cropped_data_tensor = torch.stack(cropped_data)
    cropped_masks_tensor = torch.stack(cropped_masks)

    cropped_masks_tensor = cropped_masks_tensor.unsqueeze(1)

    # Print the shapes of the resulting tensors
    #print("Shape of input data: ",cropped_data_tensor.shape)
    #print("Shape of mask data: ",cropped_masks_tensor.shape)

    dataset = TensorDataset(torch.Tensor(cropped_data_tensor), torch.Tensor(cropped_masks_tensor) )

    return dataset


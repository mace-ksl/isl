import os
import yaml
import data_set
import random
import torch
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
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

def crop_dataset(input,output, width=256,height=256):
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

    print(input.shape,output.shape)
    # Perform random cropping for each image
    for i in range(len(input)):
        # Apply random crop transform to both the data and the mask
        for j in range(num_patches_per_image):

            ok_crop = False
            while(ok_crop == False):
                cropped_data_patch,cropped_mask_patch = randomCrop(input[i], output[i],width=width,height=height)
                
                normalized_mask = (cropped_mask_patch - cropped_mask_patch.min()) / (cropped_mask_patch.max() - cropped_mask_patch.min())
                print(normalized_mask.shape)
                # Assuming normalized_mask is a PyTorch tensor of shape [1, 256, 256] and already normalized
                binary_mask = (normalized_mask > 0.5).squeeze()  # Convert to binary mask and remove channel dimension if exists

                # Find connected components
                labeled_mask, num_features = ndimage.label(binary_mask)  # Convert tensor to numpy array and label components

                # Find sizes of connected components
                component_sizes = ndimage.sum(binary_mask, labeled_mask, range(num_features + 1))

                # Check if any component is larger than 1000
                ok = np.any(component_sizes >= 6000)

                binary_mask2 = (normalized_mask < 0.01).squeeze()
                labeled_mask, num_features = ndimage.label(binary_mask2)  # Convert tensor to numpy array and label components
                component_sizes = ndimage.sum(binary_mask2, labeled_mask, range(num_features + 1))
                ok2 = np.any(component_sizes >= 4000)
                if ok and ok2:
                    normalized_data = (cropped_data_patch - cropped_data_patch.min()) / (cropped_data_patch.max() - cropped_data_patch.min())
                    print(normalized_data.shape,normalized_mask.shape)

                    
                    image1_channel1 = normalized_data[0]  # First image, corresponding to the first channel
                    image2_channel1 = normalized_data[1]  # Second image, corresponding to the second channel
                    image3_channel1 = normalized_data[2]  # Third image, corresponding to the third channel

                    image1_channel1_pil = Image.fromarray((image1_channel1 * 255).astype(np.uint8))
                    image2_channel1_pil = Image.fromarray((image2_channel1 * 255).astype(np.uint8))
                    image3_channel1_pil = Image.fromarray((image3_channel1 * 255).astype(np.uint8))

                    image2_np = normalized_mask.squeeze()

                    image2_pil = Image.fromarray((image2_np * 255).astype(np.uint8))
                    count += 1
                    prefix = "\\"+str(count)
                    image1_channel1_pil.save(path+prefix+r"_bf_ch_1.tiff")
                    image2_channel1_pil.save(path+prefix+r"_bf_ch_2.tiff")
                    image3_channel1_pil.save(path+prefix+r"_pc.tiff")
                    image2_pil.save(path+prefix+r"_dapi.png")
                    ok_crop = True
                    


# Set current path
current_path = os.getcwd()

# Read parameters from config.yaml
with open(os.path.join(current_path,'config.yaml'), 'r') as file:
    config = yaml.safe_load(file)

split_name = config['split_name']
crop_height = config['crop_height']
crop_width = config['crop_width']

# Create data set object
components = current_path.split(os.path.sep)
data_set_path= components[0] + os.path.sep + os.path.join(*components[1:-1], 'data_set')
data = data_set.DataSet(data_set_path)


data_train_in = data.get_input_images_as_array(split_name,"train")
data_train_out = data.get_output_images_as_array(split_name,"train")
crop_dataset(data_train_in,data_train_out)

data_val_in = data.get_input_images_as_array(split_name,"val")
data_val_out = data.get_output_images_as_array(split_name,"val")
crop_dataset(data_val_in,data_val_out)
data_test_in = data.get_input_images_as_array(split_name,"test")
data_test_out = data.get_output_images_as_array(split_name,"test")
crop_dataset(data_test_in,data_test_out)

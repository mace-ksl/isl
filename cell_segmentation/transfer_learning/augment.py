import random
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader,TensorDataset

def randomCrop(img, mask, width=224, height=224):
    assert img.shape[1] >= height
    assert img.shape[2] >= width
    assert img.shape[1] == mask.shape[1]
    assert img.shape[2] == mask.shape[2]

    x = random.randint(0, img.shape[2] - width)
    y = random.randint(0, img.shape[1] - height)

    img = img[:, y:y+height, x:x+width]
    mask = mask[:, y:y+height, x:x+width]

    return img, mask

def crop_dataset(dataset, width=224,height=224):
    cropped_data = []
    cropped_masks = []

    # Number of patches to extract per image
    num_patches_per_image = 5

    # Perform random cropping for each image
    for i in range(len(dataset)):
        # Apply random crop transform to both the data and the mask
        for _ in range(num_patches_per_image):

            #print(dataset[i][0].shape)
            #print(dataset[i][1].shape)
            ok_crop = False
            while(ok_crop == False):
                cropped_data_patch,cropped_mask_patch = randomCrop(dataset[i][0], dataset[i][1],width=width,height=height)
                # Cells that are big enough
                if (cropped_mask_patch[0, :, :].reshape(-1) != 0).sum() > 15000:
                    ok_crop = True
            #print("---------",cropped_data_patch.shape,cropped_mask_patch.shape)
            #print(cropped_data_patch.shape)
            # Append the cropped patches to the lists
            cropped_data.append(cropped_data_patch.squeeze(0))  # Remove the batch dimension
            cropped_masks.append(cropped_mask_patch.squeeze(0))  # Remove the batch dimension

            
    # Convert the lists to tensors
    cropped_data_tensor = torch.stack(cropped_data)
    cropped_masks_tensor = torch.stack(cropped_masks)

    cropped_masks_tensor = cropped_masks_tensor.unsqueeze(1)

    # Print the shapes of the resulting tensors
    #print("Shape of input data: ",cropped_data_tensor.shape)
    #print("Shape of mask data: ",cropped_masks_tensor.shape)

    dataset = TensorDataset(torch.Tensor(cropped_data_tensor), torch.Tensor(cropped_masks_tensor) )

    return dataset


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
            cropped_data_patch,cropped_mask_patch = randomCrop(dataset[i][0], dataset[i][1],width=width,height=height)
            #print(cropped_data_patch.shape)
            # Append the cropped patches to the lists
            cropped_data.append(cropped_data_patch.squeeze(0))  # Remove the batch dimension
            cropped_masks.append(cropped_mask_patch.squeeze(0))  # Remove the batch dimension

            
    # Convert the lists to tensors
    cropped_data_tensor = torch.stack(cropped_data)
    cropped_masks_tensor = torch.stack(cropped_masks)

    cropped_masks_tensor = cropped_masks_tensor.unsqueeze(1)

    # Print the shapes of the resulting tensors
    print("Shape of input data: ",cropped_data_tensor.shape)
    print("Shape of mask data: ",cropped_masks_tensor.shape)

    dataset = TensorDataset(torch.Tensor(cropped_data_tensor), torch.Tensor(cropped_masks_tensor) )

    return dataset


def plot_random_image_from_dataset(dataset):
    random_index = torch.randint(len(dataset), size=(1,)).item()

    # Get the random image and mask

    random_image = dataset[random_index][0]
    random_mask = dataset[random_index][1]

    # Convert to numpy arrays for plotting
    image_np = random_image.cpu().numpy()
    mask_np = random_mask.cpu().numpy()

    # Plot the images and mask
    plt.figure(figsize=(10, 4))

    # Plot each channel of the image separately in grayscale
    for i in range(3):
        plt.subplot(1, 4, i + 1)
        plt.imshow(image_np[i], cmap='gray')
        plt.title(f"Channel {i + 1}")

    # Plot the combined image
    combined_image = image_np.transpose(1, 2, 0)
    plt.subplot(1, 4, 4)
    plt.imshow(mask_np.squeeze(),cmap='gray')
    plt.title("Mask")

    # Show the mask
    plt.show()

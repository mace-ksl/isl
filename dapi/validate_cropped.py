import torch
import model
import os 
import pytorch_lightning as pl 
import data_set_cropped
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import yaml
from scipy.stats import pearsonr

# Set current path
current_path = os.getcwd()

# Read parameters from config.yaml
with open(os.path.join(current_path,'config.yaml'), 'r') as file:
    config = yaml.safe_load(file)

# Save parameters as local vars
learning_rate = config['learning_rate']
batch_size = config['batch_size']
max_epochs = config['max_epochs']
crop_height = config['crop_height']
crop_width = config['crop_width']
split_name = config['split_name']

# Create data set object
components = current_path.split(os.path.sep)
data_set_path= components[0] + os.path.sep + os.path.join(*components[1:-1], 'data_set')
data = data_set_cropped.DataSet(data_set_path)

# Create train, validation and test sets as numpy arrays
data_train_in = data.get_input_images_as_array(split_name,"train")
data_train_out = data.get_output_images_as_array(split_name,"train")

data_test_in = data.get_input_images_as_array(split_name,"test")
data_test_out = data.get_output_images_as_array(split_name,"test")

# Create loaders
train_loader, test_loader = data.create_torch_data_loader(x_train=data_train_in,
                                                          y_train=data_train_out,
                                                          x_test=data_test_in,
                                                          y_test=data_test_out,
                                                          batch_size=batch_size,
                                                          height=256,
                                                          width=256)


# Load transfer learning model (better)
model_train = model.Model.load_from_checkpoint(r'E:\models\DAPI_augmented\best_model/model.ckpt', encoder_name="mit_b2" ,learning_rate=learning_rate)

# ------------------------------
# Load and test pretrained model
#model_train = model.Model(model_path=cell_mask_model_path, encoder_name="mit_b2" ,learning_rate=learning_rate)

def plot_results(image, ground_truth_array, pr_mask_array):
    plt.figure(figsize=(15, 8))

    # Display original images
    for j in range(3):  # Assuming 'image' has three channels
        plt.subplot(1, 6, j + 1)
        plt.imshow(image[:, :, j], cmap='gray')  
        plt.title(f"Image Channel {j+1}")
        plt.axis("off")

    # Display ground truth mask
    plt.subplot(1, 6, 4)
    plt.imshow(ground_truth_array, cmap='gray')
    plt.title("Ground Truth DAPI")
    plt.axis("off")
    
    # Display predicted mask
    plt.subplot(1, 6, 5)
    plt.imshow(pr_mask_array, cmap='gray')
    plt.title("Predicted DAPI")
    plt.axis("off")

    #normalized_image = (pr_mask_array - 0.5) * 1000
    min_val = np.min(pr_mask_array)
    max_val = np.max(pr_mask_array)

    # Normalize to 0-255
    normalized_image = 254 * (pr_mask_array - np.min(pr_mask_array)) / (np.max(pr_mask_array) - np.min(pr_mask_array)) + 1
    factor=0.1
    factor = max(0, min(factor, 1))
    darkened_img = np.clip(normalized_image * factor, 0, 255).astype(np.uint8)
    plt.subplot(1, 6, 6)
    plt.imshow(darkened_img, cmap='gray')
    plt.title("Predicted DAPI balanced")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_pcc(normalized_image,ground_truth_array):

    # Calculate Pearson correlation coefficient
    correlation, _ = pearsonr(normalized_image.flatten(), ground_truth_array.flatten())

    plt.figure(figsize=(8, 6))  # Set the figure size

    # Determine colors: points where predicted value is higher than ground truth are red, others are blue
    colors = ['green' if pred > gt else 'blue' for pred, gt in zip(normalized_image.flatten(), ground_truth_array.flatten())]
    colors = ['skyblue']
    print(normalized_image.flatten().shape)
    # Create scatter plot
    plt.scatter(normalized_image.flatten(), ground_truth_array.flatten(), color=colors, alpha=0.5)
    plt.plot([min(normalized_image.min(), ground_truth_array.min()), max(normalized_image.max(), ground_truth_array.max())], 
         [min(normalized_image.min(), ground_truth_array.min()), max(normalized_image.max(), ground_truth_array.max())], 
         'r--', linewidth=2, label=f"Pearson: {correlation:.2f}")


    plt.xlabel("Predicted DAPI")
    plt.ylabel("Ground Truth DAPI")

    plt.legend()
    plt.show()


# Evaluate 200 images from the testdata set
# Calculate the PCC and plot the results
# Best result 0.93±0.03

num_batches_to_visualize = 200 
images_per_batch = 1 
all_pcc = []
print(f"Number of batches: {len(test_loader)}")
print(f"Number of samples: {len(test_loader.dataset)}")

for batch_index, batch in enumerate(test_loader):
    if batch_index >= num_batches_to_visualize:
        break
    print(f"Batch number: {batch_index}")
    with torch.no_grad():
        model_train.eval()  
        logits = model_train(batch[0]) 
    pr_masks = logits.sigmoid()  

    # Zip together the images, ground truth masks, and predicted masks
    for i, (image, gt_mask, pr_mask) in enumerate(zip(batch[0], batch[1], pr_masks)):
        if i >= images_per_batch:
            break  # Display only a limited number of images per batch
        print(f"Image: {i}")
        
        #print(pr_mask.numpy().dtype, gt_mask.numpy().dtype)
        #print(image.shape)
        #print(gt_mask.shape)
        #print(pr_mask.shape)
        image = image.numpy().transpose(1, 2, 0)  # Adjust dimensions for plotting if necessary
        ground_truth_array = gt_mask.numpy().squeeze(0)
        pr_mask_array = pr_mask.numpy().squeeze(0)
        
        plot_results(image, ground_truth_array, pr_mask_array)

        min_val = np.min(pr_mask_array)
        max_val = np.max(pr_mask_array)
        # Normalize to 0-255
        normalized_pr_mask = (pr_mask_array - min_val) / (max_val - min_val) * 255
        normalized_image = normalized_pr_mask.astype(np.uint8)  # Convert to unsigned byte type
        correlation, _ = pearsonr(normalized_image.flatten(), ground_truth_array.flatten())
        all_pcc.append(correlation)
        #plot_pcc(normalized_image,ground_truth_array)

        
plt.figure(figsize=(8, 6))
plt.hist(all_pcc, bins=10, color='skyblue', edgecolor='black') 

plt.xlabel('Pearson Correlation Coefficient (PCC)')
plt.ylabel('Number of PCC scores')

mean_corr = np.mean(all_pcc)
std_corr = np.std(all_pcc)

print(f"{mean_corr:.2f}±{std_corr:.2f}")

plt.show()
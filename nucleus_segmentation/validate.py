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
import data_set
import plot

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
data = data_set.DataSet(data_set_path)

# Create train, validation and test sets as numpy arrays
data_train_in = data.get_input_images_as_array(split_name,"train")
data_train_out = data.get_output_images_as_array(split_name,"train")

data_test_in = data.get_input_images_as_array(split_name,"test")
data_test_out = data.get_output_images_as_array(split_name,"test")

#plot.plot_random_set_of_images(data_train_in,data_train_out,number_plots=5)
# Create loaders
train_loader, test_loader = data.create_torch_data_loader(x_train=data_train_in,
                                                          y_train=data_train_out,
                                                          x_test=data_test_in,
                                                          y_test=data_test_out,
                                                          batch_size=batch_size,
                                                          height=256,
                                                          width=256)

#model_train = model.Model(model_path=cell_mask_model_path, encoder_name="mit_b2" ,learning_rate=learning_rate)
#model_train = model_train.load_from_checkpoint(r'E:\Data_sets\Github\timm\isl\data_set/model.ckpt',model_path=cell_mask_model_path, encoder_name="mit_b2" ,learning_rate=learning_rate)

# Load transfer learning model (better)
# First way:
#model_train = model.Model.load_from_checkpoint(r'C:\Users\Marcel\Desktop\models\TRANSFERGOOOOOOOOOOOOOOD\low_training/model.ckpt',model_path=cell_mask_model_path, encoder_name="mit_b2" ,learning_rate=learning_rate)
#model_train = model.Model.load_from_checkpoint(r'C:\Users\Marcel\Desktop\models/model.ckpt',model_path=cell_mask_model_path, encoder_name="mit_b2" ,learning_rate=learning_rate)
path = r'E:\models\NUCLEUS_SEGMENTATION\model_6\model.ckpt'
model_train = model.Model.load_from_checkpoint(path,model_path=path, encoder_name="mit_b2" ,learning_rate=learning_rate)
# Second way:
#model_train = model.Model(model_path=cell_mask_model_path, encoder_name="mit_b2" ,learning_rate=learning_rate)
#model_train = model_train.load_from_checkpoint(r'E:\Data_sets\Github\timm\isl\data_set/model.ckpt',model_path=cell_mask_model_path, encoder_name="mit_b2" ,learning_rate=learning_rate)

# ------------------------------
# Load and test pretrained model
#model_train = model.Model(model_path=cell_mask_model_path, encoder_name="mit_b2" ,learning_rate=learning_rate)

# ------------------------------
#trainer = pl.Trainer(accelerator='gpu', devices=1,num_nodes=1, max_epochs=1, default_root_dir = data.data_dir)

def crop_to_2048(tensor):
    # Assuming tensor dimensions are at least 2048x2048
    _, height, width = tensor.shape
    start_y = (height - 2048) // 2
    start_x = (width - 2048) // 2

    # Perform the crop
    cropped_tensor = tensor[:, start_y:start_y + 2048, start_x:start_x + 2048]
    return cropped_tensor

def plot_results(image, ground_truth_array, pr_mask_array):
    plt.figure(figsize=(15, 8))

    # Display original images
    for j in range(3):  # Assuming 'image' has three channels
        plt.subplot(1, 7, j + 1)
        plt.imshow(image[:, :, j], cmap='gray')  
        plt.title(f"Image Channel {j+1}")
        plt.axis("off")

    # Display ground truth mask
    plt.subplot(1, 7, 4)
    plt.imshow(ground_truth_array, cmap='gray')
    plt.title("Ground Truth DAPI")
    plt.axis("off")
    
    # Display predicted mask
    plt.subplot(1, 7, 5)
    plt.imshow(pr_mask_array, cmap='gray')
    plt.title("Predicted DAPI")
    plt.axis("off")

    #normalized_image = (pr_mask_array - 0.5) * 1000
    min_val = np.min(pr_mask_array)
    max_val = np.max(pr_mask_array)

    # Normalize to 0-255
    normalized_pr_mask = (pr_mask_array - min_val) / (max_val - min_val) * 255
    normalized_image = normalized_pr_mask.astype(np.uint8)  # Convert to unsigned byte type

    plt.subplot(1, 7, 6)
    plt.imshow(normalized_image, cmap='gray')
    plt.title("Predicted DAPI balanced")
    plt.axis("off")

    thresholded_mask = (pr_mask_array >= 0.5)
    plt.subplot(1, 7, 7)
    plt.imshow(thresholded_mask, cmap='gray')
    plt.title("Predicted DAPI balanced")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


def plot_pcc(normalized_image,ground_truth_array):

    # Calculate Pearson correlation coefficient
    correlation, _ = pearsonr(normalized_image.flatten(), ground_truth_array.flatten())

    plt.figure(figsize=(8, 6))  # Set the figure size

    # Determine colors: points where predicted value is higher than ground truth are red, others are blue
    colors = ['red' if pred > gt else 'blue' for pred, gt in zip(normalized_image.flatten(), ground_truth_array.flatten())]

    # Create scatter plot
    plt.scatter(normalized_image.flatten(), ground_truth_array.flatten(), color=colors, alpha=0.5, label=f"Pearson: {correlation:.2f}")
    plt.plot([min(normalized_image.min(), ground_truth_array.min()), max(normalized_image.max(), ground_truth_array.max())], 
            [min(normalized_image.min(), ground_truth_array.min()), max(normalized_image.max(), ground_truth_array.max())], 
            'r--', linewidth=2)  # Red dashed line showing the line of perfect correlation

    # Label the axes and add title and legend
    plt.xlabel("Predicted DAPI (normalized)")
    plt.ylabel("Ground Truth DAPI")
    plt.title("Correlation Plot between Predicted and Ground Truth DAPI")
    plt.legend()

    plt.show()

num_batches_to_visualize = 200 
images_per_batch = 1 

all_pcc = []
print(f"Number of batches: {len(test_loader)}")
print(f"Number of samples: {len(test_loader.dataset)}")
# Iterate over the test loader
for batch_index, batch in enumerate(test_loader):
    if batch_index >= num_batches_to_visualize:
        break
    print(f"Number of image: {batch_index}")
    with torch.no_grad():
        model_train.eval()  # Set the model to evaluation mode
        cropped_images = torch.stack([crop_to_2048(img) for img in batch[0]])
        logits = model_train(cropped_images)
    pr_masks = logits.sigmoid()  # Apply sigmoid to convert logits to probabilities

    # Zip together the images, ground truth masks, and predicted masks
    for i, (image, gt_mask, pr_mask) in enumerate(zip(batch[0], batch[1], pr_masks)):
        if i >= images_per_batch:
            break  # Display only a limited number of images per batch
        #print(f"Image: {i}")
        
        #print(pr_mask.numpy().dtype, gt_mask.numpy().dtype)
        #print(image.shape)
        #print(gt_mask.shape)
        #print(pr_mask.shape)
        gt_mask_np = crop_to_2048(gt_mask).numpy().squeeze(0)  # Crop GT mask
        pr_mask_np = crop_to_2048(pr_mask).numpy().squeeze(0)  # Crop predicted mask

        image = image.numpy().transpose(1, 2, 0)  # Adjust dimensions for plotting if necessary
        ground_truth_array = gt_mask_np
        pr_mask_array = pr_mask_np
        
        #plot_results(image, ground_truth_array, pr_mask_array)

        min_val = np.min(pr_mask_array)
        max_val = np.max(pr_mask_array)
        # Normalize to 0-255
        normalized_pr_mask = (pr_mask_array - min_val) / (max_val - min_val) * 255
        normalized_image = normalized_pr_mask.astype(np.uint8)  # Convert to unsigned byte type
        print(normalized_image.shape,ground_truth_array.shape)
        correlation, _ = pearsonr(normalized_image.flatten(), ground_truth_array.flatten())

        #plot_pcc(normalized_image,ground_truth_array)

        all_pcc.append(correlation)


plt.figure(figsize=(8, 6))
plt.hist(all_pcc, bins=10, color='skyblue', edgecolor='black') 

plt.xlabel('Pearson Correlation Coefficient (PCC)')
plt.ylabel('Number of PCC scores')

mean_corr = np.mean(all_pcc)
std_corr = np.std(all_pcc)

print(f"{mean_corr:.2f}±{std_corr:.2f}")

plt.show()
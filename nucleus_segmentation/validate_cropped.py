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

import matplotlib.colors as mcolors
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
print(data_test_out.shape)
# Create loaders
train_loader, test_loader = data.create_torch_data_loader(x_train=data_train_in,
                                                          y_train=data_train_out,
                                                          x_test=data_test_in,
                                                          y_test=data_test_out,
                                                          batch_size=batch_size,
                                                          height=256,
                                                          width=256)
print(train_loader)
print("-----------------------------")
path = r'E:\models\NUCLEUS_SEGMENTATION_augmented\best_model\model.ckpt'
model_train = model.Model.load_from_checkpoint(path,model_path=path, encoder_name="mit_b2" ,learning_rate=learning_rate)


def calculate_iou(ground_truth, prediction):
    
    ground_truth = ground_truth.astype(np.bool_)
    prediction = prediction.astype(np.bool_)

    intersection = np.logical_and(ground_truth, prediction)
    union = np.logical_or(ground_truth, prediction)

    union_sum = np.sum(union)
    if union_sum == 0:
        return 0
    
    iou = np.sum(intersection) / union_sum
    return iou

def plot_masks_only(ground_truth_array, pr_mask_array,iou):
    """
    Plot results without bf and pc images. Only masks
    """
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(ground_truth_array, cmap='gray')
    plt.title("Ground Truth DAPI")
    plt.axis("off")
    
    plt.subplot(1, 3, 2)
    plt.imshow(pr_mask_array, cmap='gray')
    plt.title("Predicted DAPI")
    plt.axis("off")

    #normalized_image = (pr_mask_array - 0.5) * 1000
    min_val = np.min(pr_mask_array)
    max_val = np.max(pr_mask_array)

    # Normalize to 0-255
    normalized_pr_mask = (pr_mask_array - min_val) / (max_val - min_val) * 255
    normalized_image = normalized_pr_mask.astype(np.uint8) 

    true_positives = (normalized_image == 255) & (ground_truth_array == 1)
    true_negatives = (normalized_image == 0) & (ground_truth_array == 0)
    false_positives = (normalized_image == 255) & (ground_truth_array == 0)
    false_negatives = (normalized_image == 0) & (ground_truth_array == 1)

    overlay_image = np.zeros((ground_truth_array.shape[0], ground_truth_array.shape[1], 3), dtype=np.float32)  # RGB
    overlay_image[true_positives] = [1, 1, 1]  # White for true positives
    overlay_image[false_positives] = [1, 0, 0]  # Red for false positives
    overlay_image[false_negatives] = [0, 0, 1]  # Blue for false negatives

    plt.subplot(1, 3, 3)
    plt.imshow(ground_truth_array, cmap='gray', interpolation='nearest')  # Display ground truth in grayscale
    plt.imshow(overlay_image, interpolation='nearest')  # Overlay the color-coded prediction results
    plt.axis('off') 
    plt.title(f'Mask Comparison\nIoU: {iou:.2f}')  # Include IoU in the title

    plt.tight_layout()
    plt.show()

def plot_results(image, ground_truth_array, pr_mask_array,iou):
    """
    Plot results with bf and pc images
    """
    plt.figure(figsize=(15, 8))

    for j in range(3):  
        plt.subplot(1, 6, j + 1)
        plt.imshow(image[:, :, j], cmap='gray')  
        plt.title(f"Image Channel {j+1}")
        plt.axis("off")

    plt.subplot(1, 6, 4)
    plt.imshow(ground_truth_array, cmap='gray')
    plt.title("Ground Truth DAPI")
    plt.axis("off")
    
    plt.subplot(1, 6, 5)
    plt.imshow(pr_mask_array, cmap='gray')
    plt.title("Predicted DAPI")
    plt.axis("off")

    min_val = np.min(pr_mask_array)
    max_val = np.max(pr_mask_array)

    # Normalize to 0-255
    normalized_pr_mask = (pr_mask_array - min_val) / (max_val - min_val) * 255
    normalized_image = normalized_pr_mask.astype(np.uint8) 

    reds = plt.cm.get_cmap('summer', 256)
    
    new_colors = reds(np.linspace(0, 1, 256))
    new_colors[0, :] = [0, 0, 0, 1]  
    #print(normalized_image,ground_truth_array)

    print(np.max(normalized_image),np.max(ground_truth_array))

    true_positives = (normalized_image == 255) & (ground_truth_array == 1)
    true_negatives = (normalized_image == 0) & (ground_truth_array == 0)
    false_positives = (normalized_image == 255) & (ground_truth_array == 0)
    false_negatives = (normalized_image == 0) & (ground_truth_array == 1)

    # Create RGB overlay image
    overlay_image = np.zeros((ground_truth_array.shape[0], ground_truth_array.shape[1], 3), dtype=np.float32)  # RGB
    overlay_image[true_positives] = [1, 1, 1]  
    overlay_image[false_positives] = [1, 0, 0]  
    overlay_image[false_negatives] = [0, 0, 1] 

    plt.subplot(1, 6, 6)
    plt.imshow(ground_truth_array, cmap='gray', interpolation='nearest')  
    plt.imshow(overlay_image, interpolation='nearest')  
    plt.title('Prediction Analysis')
    plt.axis('off')  
    plt.title(f'Mask Comparison\nIoU: {iou:.2f}') 

    plt.tight_layout()
    plt.show()


def plot_pcc(normalized_image,ground_truth_array):
    """
    Calculate and plot the pearson correlation coefficient
    """
    correlation, _ = pearsonr(normalized_image.flatten(), ground_truth_array.flatten())

    plt.figure(figsize=(8, 6))  

    colors = ['red' if pred > gt else 'blue' for pred, gt in zip(normalized_image.flatten(), ground_truth_array.flatten())]

    plt.scatter(normalized_image.flatten(), ground_truth_array.flatten(), color=colors, alpha=0.5, label=f"Pearson: {correlation:.2f}")
    plt.plot([min(normalized_image.min(), ground_truth_array.min()), max(normalized_image.max(), ground_truth_array.max())], 
            [min(normalized_image.min(), ground_truth_array.min()), max(normalized_image.max(), ground_truth_array.max())], 
            'r--', linewidth=2)  

    plt.xlabel("Predicted DAPI (normalized)")
    plt.ylabel("Ground Truth DAPI")
    plt.title("Correlation Plot between Predicted and Ground Truth DAPI")
    plt.legend()

    plt.show()

num_batches_to_visualize = 200 
images_per_batch = 1 
all_iou=[]
all_pcc = []
print(f"Number of batches: {len(test_loader)}")
for batch_index, batch in enumerate(test_loader):
    if batch_index >= num_batches_to_visualize:
        break
    print(f"Number of image: {batch_index}")
    with torch.no_grad():
        model_train.eval() 
        logits = model_train(batch[0]) 
    pr_masks = logits.sigmoid() 

    
    for i, (image, gt_mask, pr_mask) in enumerate(zip(batch[0], batch[1], pr_masks)):
        if i >= images_per_batch:
            break  
        #print(f"Image: {i}")
        #print(pr_mask.numpy().dtype, gt_mask.numpy().dtype)
        #print(image.shape)
        #print(gt_mask.shape)
        #print(pr_mask.shape)

        image = image.numpy().transpose(1, 2, 0)  
        ground_truth_array = gt_mask.numpy().squeeze(0)
        pr_mask_array = pr_mask.numpy().squeeze(0)

        threshold = 0.5
        pr_mask_array = (pr_mask_array > threshold).astype(int)
        ground_truth_array = (ground_truth_array > threshold).astype(int)

        min_val = np.min(pr_mask_array)
        max_val = np.max(pr_mask_array)

        # Normalize to 0-255
        normalized_pr_mask = (pr_mask_array - min_val) / (max_val - min_val) * 255
        normalized_image = normalized_pr_mask.astype(np.uint8)  # Convert to unsigned byte type

        correlation, _ = pearsonr(normalized_image.flatten(), ground_truth_array.flatten())
        #print(normalized_image)
        iou = calculate_iou(ground_truth_array,pr_mask_array)
       
        plot_results(image, ground_truth_array, pr_mask_array,iou)

        #plot_masks_only(ground_truth_array, pr_mask_array,iou)
        all_iou.append(iou)
        #print(iou)
        #plot_pcc(normalized_image,ground_truth_array)
        all_pcc.append(correlation)


# Plot histogram of IoU
mean_corr = np.mean(all_iou)
std_corr = np.std(all_iou)

print(f"{mean_corr:.2f}±{std_corr:.2f}")

plt.figure(figsize=(8, 6))
plt.hist(all_iou, bins=10, color='skyblue', edgecolor='black') 

plt.xlabel('Intersection over Union (IoU)')
plt.ylabel('Number of IoU scores')

plt.show()

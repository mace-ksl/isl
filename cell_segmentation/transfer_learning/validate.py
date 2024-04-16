import torch
import model
import os 
import pytorch_lightning as pl 
import data_set
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
import yaml

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
cell_mask_model_path = config['cell_mask_model_path']

# Create data set object
components = current_path.split(os.path.sep)
data_set_path= components[0] + os.path.sep + os.path.join(*components[1:-2], 'data_set')
data = data_set.DataSet(data_set_path)

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

print(f"Load: {cell_mask_model_path}")
#model_train = model.Model(model_path=cell_mask_model_path, encoder_name="mit_b2" ,learning_rate=learning_rate)
#model_train = model_train.load_from_checkpoint(r'E:\Data_sets\Github\timm\isl\data_set/model.ckpt',model_path=cell_mask_model_path, encoder_name="mit_b2" ,learning_rate=learning_rate)

# Load transfer learning model (better)
# First way:
#model_train = model.Model.load_from_checkpoint(r'C:\Users\Marcel\Desktop\models/model.ckpt',model_path=cell_mask_model_path, encoder_name="mit_b2" ,learning_rate=learning_rate)
model_train = model.Model.load_from_checkpoint(r'E:\models\TripleLoss/model.ckpt',model_path=cell_mask_model_path, encoder_name="mit_b2" ,learning_rate=learning_rate)
# Second way:
#model_train = model.Model(model_path=cell_mask_model_path, encoder_name="mit_b2" ,learning_rate=learning_rate)
#model_train = model_train.load_from_checkpoint(r'E:\Data_sets\Github\timm\isl\data_set/model.ckpt',model_path=cell_mask_model_path, encoder_name="mit_b2" ,learning_rate=learning_rate)

# ------------------------------
# Load and test pretrained model
#model_train = model.Model(model_path=cell_mask_model_path, encoder_name="mit_b2" ,learning_rate=learning_rate)

# ------------------------------
#trainer = pl.Trainer(accelerator='gpu', devices=1,num_nodes=1, max_epochs=1, default_root_dir = data.data_dir)


num_batches_to_visualize = 20 
images_per_batch = 1 

for batch_index, batch in enumerate(test_loader):
    if batch_index >= num_batches_to_visualize:
        break
    print(f"Batch number: {batch_index}")
    with torch.no_grad():
        model_train.eval()  # Set the model to evaluation mode
        logits = model_train(batch[0])  # Get model output for the batch
    pr_masks = logits.sigmoid()  # Apply sigmoid to convert logits to probabilities

    # Zip together the images, ground truth masks, and predicted masks
    for i, (image, gt_mask, pr_mask) in enumerate(zip(batch[0], batch[1], pr_masks)):
        if i >= images_per_batch:
            break  # Display only a limited number of images per batch
        
        #image = image.numpy().transpose(1, 2, 0)  # Adjust dimensions for plotting if necessary
        #ground_truth_array = gt_mask.numpy().squeeze(0)
        #pr_mask_array = pr_mask.numpy().squeeze(0)

        plt.figure(figsize=(15, 8))
        print(pr_mask.numpy().dtype,gt_mask.numpy().dtype)
        print(image.shape)
        print(gt_mask.shape)
        print(pr_mask.shape)
        image = image.numpy()
        for i in range(3):
            plt.subplot(1, 8, i + 1)
            plt.imshow(image[i], cmap='gray')
            plt.title(f"Image {i+1}")
            plt.axis("off")

        ground_truth_array = gt_mask.numpy().squeeze(0)   
        #print(ground_truth_array.shape)
        plt.subplot(1, 8, 4)
        plt.imshow(ground_truth_array[0], cmap='gray')
        plt.title("Ground truth topology")
        plt.axis("off")

        plt.subplot(1, 8, 5)
        plt.imshow(ground_truth_array[1], cmap='gray')
        plt.title("Ground truth semantic")
        plt.axis("off")

        pr_mask_array = pr_mask.numpy()
        
        thresholded_array = (pr_mask_array[0] <= 0.5).astype(int)
        plt.subplot(1, 8, 6)
        plt.imshow(thresholded_array,cmap='gray')
        plt.title("Mask semantic")
        plt.axis("off")

        plt.subplot(1, 8, 7)
        plt.imshow(pr_mask_array[1],cmap='gray')
        plt.title("Mask topology")
        plt.axis("off")

        print(pr_mask_array[1].shape)
        pr_mask[1][pr_mask_array[1] > 0.5] = 1
        plt.subplot(1, 8, 8)
        plt.imshow(pr_mask_array[1],cmap='gray')
        plt.title("Mask semantic")
        plt.axis("off")
        
        plt.tight_layout() 
        plt.show()
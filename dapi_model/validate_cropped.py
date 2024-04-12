import torch
import model
import os 
import pytorch_lightning as pl 
import data_set_cropped
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

#model_train = model.Model(model_path=cell_mask_model_path, encoder_name="mit_b2" ,learning_rate=learning_rate)
#model_train = model_train.load_from_checkpoint(r'E:\Data_sets\Github\timm\isl\data_set/model.ckpt',model_path=cell_mask_model_path, encoder_name="mit_b2" ,learning_rate=learning_rate)

# Load transfer learning model (better)
# First way:
#model_train = model.Model.load_from_checkpoint(r'C:\Users\Marcel\Desktop\models\TRANSFERGOOOOOOOOOOOOOOD\low_training/model.ckpt',model_path=cell_mask_model_path, encoder_name="mit_b2" ,learning_rate=learning_rate)
#model_train = model.Model.load_from_checkpoint(r'C:\Users\Marcel\Desktop\models/model.ckpt',model_path=cell_mask_model_path, encoder_name="mit_b2" ,learning_rate=learning_rate)
model_train = model.Model.load_from_checkpoint('E:/isl/data_set/model.ckpt', encoder_name="mit_b2" ,learning_rate=learning_rate)
# Second way:
#model_train = model.Model(model_path=cell_mask_model_path, encoder_name="mit_b2" ,learning_rate=learning_rate)
#model_train = model_train.load_from_checkpoint(r'E:\Data_sets\Github\timm\isl\data_set/model.ckpt',model_path=cell_mask_model_path, encoder_name="mit_b2" ,learning_rate=learning_rate)

# ------------------------------
# Load and test pretrained model
#model_train = model.Model(model_path=cell_mask_model_path, encoder_name="mit_b2" ,learning_rate=learning_rate)

# ------------------------------
#trainer = pl.Trainer(accelerator='gpu', devices=1,num_nodes=1, max_epochs=1, default_root_dir = data.data_dir)

num_batches_to_visualize = 5  # Set the number of batches you want to visualize
images_per_batch = 5  # Max number of images per batch to display

# Iterate over the test loader
for batch_index, batch in enumerate(test_loader):
    if batch_index >= num_batches_to_visualize:
        break

    with torch.no_grad():
        model_train.eval()  # Set the model to evaluation mode
        logits = model_train(batch[0])  # Get model output for the batch
    pr_masks = logits.sigmoid()  # Apply sigmoid to convert logits to probabilities

    # Zip together the images, ground truth masks, and predicted masks
    for i, (image, gt_mask, pr_mask) in enumerate(zip(batch[0], batch[1], pr_masks)):
        if i >= images_per_batch:
            break  # Display only a limited number of images per batch

        plt.figure(figsize=(15, 8))

        # Print out types and shapes for debugging
        print(pr_mask.numpy().dtype, gt_mask.numpy().dtype)
        print(image.shape)
        print(gt_mask.shape)
        print(pr_mask.shape)

        image = image.numpy().transpose(1, 2, 0)  # Adjust dimensions for plotting if necessary
        ground_truth_array = gt_mask.numpy().squeeze(0)
        pr_mask_array = pr_mask.numpy().squeeze(0)

        # Display original images
        for j in range(3):  # Assuming 'image' has three channels
            plt.subplot(1, 7, j + 1)
            plt.imshow(image[:, :, j], cmap='gray')  # Adjust indexing if your data is formatted differently
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


        pr_mask_array = pr_mask_array - pr_mask_array.min()
        pr_mask_array = pr_mask_array / pr_mask_array.max()

        # Apply gamma correction to darken the image
        gamma = 0.5  # Gamma < 1 will darken the image
        darker_image = np.power(pr_mask_array, gamma)

        plt.subplot(1, 7, 6)
        plt.imshow(darker_image, cmap='gray')
        plt.title("Predicted DAPI balanced")
        plt.axis("off")

        min_val = pr_mask_array.min()
        normalized_image = pr_mask_array - min_val
        plt.subplot(1, 7, 7)
        plt.imshow(normalized_image, cmap='gray')
        plt.title("Predicted DAPI balanced")
        plt.axis("off")

        plt.tight_layout()
        plt.show()


# Validate model
    
#valid_metrics = trainer.validate(model_train, dataloaders=val_loader, verbose=False)
#pprint(valid_metrics)
#print(type(valid_metrics))
#test_metrics = trainer.test(model_train, dataloaders=test_loader, verbose=False)
#pprint(test_metrics)
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

# Create data set object
components = current_path.split(os.path.sep)
data_set_path= components[0] + os.path.sep + os.path.join(*components[1:-1], 'data_set')
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


def crop_to_2048(tensor):
    # Assuming tensor dimensions are at least 2048x2048
    _, height, width = tensor.shape
    start_y = (height - 2048) // 2
    start_x = (width - 2048) // 2

    # Perform the crop
    cropped_tensor = tensor[:, start_y:start_y + 2048, start_x:start_x + 2048]
    return cropped_tensor
# Constants
num_batches_to_visualize = 5
images_per_batch = 5

# Iterate over the test loader
for batch_index, batch in enumerate(test_loader):
    if batch_index >= num_batches_to_visualize:
        break

    with torch.no_grad():
        model_train.eval()  # Set the model to evaluation mode
        
        # Crop and prepare images for model input
        cropped_images = torch.stack([crop_to_2048(img) for img in batch[0]])
        logits = model_train(cropped_images)
        pr_masks = logits.sigmoid()  # Convert logits to probabilities

    # Visualize each image in the batch
    for i, (cropped_image, gt_mask, pr_mask) in enumerate(zip(cropped_images, batch[1], pr_masks)):
        if i >= images_per_batch:
            break  # Limit the number of images displayed per batch

        plt.figure(figsize=(15, 8))

        # Convert tensors for plotting
        image_np = cropped_image.numpy().transpose(1, 2, 0)
        gt_mask_np = crop_to_2048(gt_mask).numpy().squeeze(0)  # Crop GT mask
        pr_mask_np = crop_to_2048(pr_mask).numpy().squeeze(0)  # Crop predicted mask

        # Display original images
        for j in range(3):  # Assuming 'image' has three channels
            plt.subplot(1, 5, j + 1)
            plt.imshow(image_np[:, :, j], cmap='gray')
            plt.title(f"Image Channel {j+1}")
            plt.axis("off")

        # Display ground truth mask
        plt.subplot(1, 5, 4)
        plt.imshow(gt_mask_np, cmap='gray')
        plt.title("Ground Truth DAPI")
        plt.axis("off")

        # Display predicted mask
        plt.subplot(1, 5, 5)
        plt.imshow(pr_mask_np, cmap='gray_r')
        plt.title("Predicted DAPI")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

# Validate model
    
#valid_metrics = trainer.validate(model_train, dataloaders=val_loader, verbose=False)
#pprint(valid_metrics)
#print(type(valid_metrics))
#test_metrics = trainer.test(model_train, dataloaders=test_loader, verbose=False)
#pprint(test_metrics)
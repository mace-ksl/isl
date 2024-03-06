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

data_train_in = data.get_input_images_as_array("split_cy5","train")
data_train_out = data.get_output_images_as_array("split_cy5","train")

data_test_in = data.get_input_images_as_array("split_cy5","test")
data_test_out = data.get_output_images_as_array("split_cy5","test")

# Create loaders
train_loader, test_loader = data.create_torch_data_loader(x_train=data_train_in,
                                                          y_train=data_train_out,
                                                          x_test=data_test_in,
                                                          y_test=data_test_out,
                                                          batch_size=batch_size,
                                                          height=256,
                                                          width=256)

model_train = model.Model.load_from_checkpoint(r'C:\Users\Marcel\Desktop\models/model.ckpt', encoder_name="mit_b2" )

trainer = pl.Trainer(accelerator='gpu', devices=1,num_nodes=1, max_epochs=1, default_root_dir = data.data_dir)



# Result visualization
batch = next(iter(test_loader))
with torch.no_grad():
    model_train.eval()
    logits = model_train(batch[0])
pr_masks = logits.sigmoid()


print("Batch: ",len(batch))

for image, gt_mask, pr_mask in zip(batch[0], batch[1], pr_masks):
    plt.figure(figsize=(12, 5))
    print(pr_mask.numpy().dtype,gt_mask.numpy().dtype)
    print(image.shape)
    print(gt_mask.shape)
    print(pr_mask.shape)
    image = image.numpy()
    for i in range(3):
        plt.subplot(1, 7, i + 1)
        plt.imshow(image[i], cmap='gray')
        plt.title(f"Image {i+1}")
        plt.axis("off")

    ground_truth_array = gt_mask.numpy().squeeze(0)   
    #print(ground_truth_array.shape)
    # Plot the first image
    plt.subplot(1, 7, 4)
    plt.imshow(ground_truth_array[0], cmap='gray')
    plt.title("Ground truth topology")
    plt.subplot(1, 7, 5)
    plt.axis("off")
    # Plot the second image
    plt.imshow(ground_truth_array[1], cmap='gray')
    plt.title("Ground truth semantic")
    plt.axis("off")
    pr_mask_array = pr_mask.numpy()
    plt.subplot(1, 7, 6)
    plt.imshow(pr_mask_array[0],cmap='gray')
    plt.title("Mask topology")
    plt.axis("off")

    pr_mask_array = pr_mask.numpy()
    plt.subplot(1, 7, 7)
    plt.imshow(pr_mask_array[1],cmap='gray')
    plt.title("Mask semantic")
    plt.axis("off")

    plt.show()

# Validate model
    
#valid_metrics = trainer.validate(model_train, dataloaders=val_loader, verbose=False)
#pprint(valid_metrics)
#print(type(valid_metrics))
#test_metrics = trainer.test(model_train, dataloaders=test_loader, verbose=False)
#pprint(test_metrics)
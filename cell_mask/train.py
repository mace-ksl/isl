import data_set_cropped
import os
import matplotlib.pyplot as plt
import model
import pytorch_lightning as pl 
from pprint import pprint
import torch
import time
import yaml
import plot

from pytorch_lightning.callbacks.early_stopping import EarlyStopping

start_time = time.time()

# Check if cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device} as training instance')

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

data_train_in = data.get_input_images_as_array(split_name,"train")
data_train_out = data.get_output_images_as_array(split_name,"train")

data_val_in = data.get_input_images_as_array(split_name,"val")
data_val_out = data.get_output_images_as_array(split_name,"val")

data_test_in = data.get_input_images_as_array(split_name,"test")
data_test_out = data.get_output_images_as_array(split_name,"test")

#print(data_test_in.shape,data_test_out.shape)
# [random_number, :, :, :]
# [number of images, 3, 256,256]
train_loader, val_loader, test_loader = data.create_torch_data_loader(data_train_in,data_train_out,data_val_in,data_val_out,data_test_in,data_test_out, batch_size=batch_size,height=crop_height,width=crop_width)

#plot.plot_random_set_of_images(data_train_in,data_train_out,number_plots=5)

model = model.Model("mit_b2",learning_rate=learning_rate)
after_forward_memory = torch.cuda.memory_allocated(device)
#print(f"GPU memory usage after forward pass: {after_forward_memory / (1024 ** 3):.2f} GB")

early_stop_callback = EarlyStopping(
  monitor='train_per_image_iou',
  patience=3,
  verbose=False,
  mode='max'
)
trainer = pl.Trainer(accelerator='gpu', devices=1,num_nodes=1, max_epochs=max_epochs, default_root_dir = data.data_dir,callbacks=early_stop_callback)

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

trainer.save_checkpoint(os.path.join(data.data_dir,"./model.ckpt"))

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Script execution time: {elapsed_time:.2f} seconds")
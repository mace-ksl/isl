import yaml
import data_set
import os
import matplotlib.pyplot as plt
import pytorch_lightning as pl 
from pprint import pprint
import torch
import time
import plot
import model

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

# Test for right dimensions in input and target data
assert data_train_in.shape[1] == 3
assert data_train_out.shape[1] == 2
assert data_test_in.shape[1] == 3
assert data_test_out.shape[1] == 2

# Plot masks for tests
#plot.plot_random_image(data_train_out)

# Create loaders
train_loader, test_loader = data.create_torch_data_loader(x_train=data_train_in,
                                                          y_train=data_train_out,
                                                          x_test=data_test_in,
                                                          y_test=data_test_out,
                                                          batch_size=batch_size,
                                                          height=256,
                                                          width=256)

# Load pretrained CellMask Model
#model_cell_mask = model.Model.load_from_checkpoint(cell_mask_model_path, encoder_name="mit_b2" )
model_cell_mask = model.Model(cell_mask_model_path,"mit_b2",learning_rate)

trainer = pl.Trainer(accelerator='gpu', devices=1,num_nodes=1, max_epochs=max_epochs, default_root_dir = data.data_dir)

trainer.fit(model_cell_mask, train_dataloaders=train_loader)

trainer.save_checkpoint(os.path.join(data.data_dir,"./model.ckpt"))

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Script execution time: {elapsed_time:.2f} seconds")
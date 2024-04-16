import torch
import model
import os 
import pytorch_lightning as pl 
import data_set

path = os.path.join(os.getcwd(),"data_set")
data = data_set.DataSet(path)
data_train_in = data.get_input_images_as_array("split_cy5","train")
data_train_out = data.get_output_images_as_array("split_cy5","train")

data_val_in = data.get_input_images_as_array("split_cy5","val")
data_val_out = data.get_output_images_as_array("split_cy5","val")

train_loader, val_loader = data.create_torch_data_loader(data_train_in,data_train_out,data_val_in,data_val_out)
model = model.Model("Unet", "mit_b2", in_channels=3, out_classes=1)

trainer = pl.Trainer(accelerator='gpu', devices=1,num_nodes=1, max_epochs=1000, default_root_dir = path)

valid_metrics = trainer.validate(model, dataloaders=val_loader, verbose=False,ckpt_path = os.path.join(path,"model.ckpt"))
print(valid_metrics)

import data_set
import os
import matplotlib.pyplot as plt
import model
import pytorch_lightning as pl 
from pprint import pprint
import torch

path = os.path.join(os.getcwd(),"data_set")
data = data_set.DataSet(path)

data_train_in = data.get_input_images_as_array("split_cy5","train")
data_train_out = data.get_output_images_as_array("split_cy5","train")

data_val_in = data.get_input_images_as_array("split_cy5","val")
data_val_out = data.get_output_images_as_array("split_cy5","val")

data_test_in = data.get_input_images_as_array("split_cy5","test")
data_test_out = data.get_output_images_as_array("split_cy5","test")

train_loader, val_loader, test_loader = data.create_torch_data_loader(data_train_in,data_train_out,data_val_in,data_val_out,data_test_in,data_test_out)

model = model.Model("Unet", "mit_b2", in_channels=3, out_classes=1)

trainer = pl.Trainer(accelerator='gpu', devices=1,num_nodes=1, max_epochs=1000, default_root_dir = data.data_dir)

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

trainer.save_checkpoint(os.path.join(data.data_dir,"./model.ckpt"))
trainer.save_checkpoint(os.path.join(data.data_dir,"./model.pt"))

# Validate 
valid_metrics = trainer.validate(model, dataloaders=val_loader, verbose=False)
pprint(valid_metrics)

# Run Test data set
test_metrics = trainer.test(model, dataloaders=test_loader, verbose=False)
pprint(test_metrics)

# Result visualization
batch = next(iter(test_loader))
with torch.no_grad():
    model.eval()
    logits = model(batch[0])
pr_masks = logits.sigmoid()

for image, gt_mask, pr_mask in zip(batch[0], batch[1], pr_masks):
    plt.figure(figsize=(10, 5))

    print(image.shape)
    print(gt_mask.shape)
    print(pr_mask.shape)
    image = image.numpy()
    for i in range(3):
        plt.subplot(1, 5, i + 1)
        plt.imshow(image[i], cmap='gray')
        plt.title(f"Image {i+1}")
        plt.axis("off")

    #plt.subplot(1, 3, 1)
    #plt.imshow(image.numpy().transpose(1, 2, 0),cmap='gray')  # convert CHW -> HWC
    #plt.title("Image")
    #plt.axis("off")

    plt.subplot(1, 5, 4)
    plt.imshow(gt_mask.numpy().squeeze(),cmap='gray') # just squeeze classes dim, because we have only one class
    plt.title("Ground truth")
    plt.axis("off")

    plt.subplot(1, 5, 5)
    plt.imshow(pr_mask.numpy().squeeze(),cmap='gray') # just squeeze classes dim, because we have only one class
    plt.title("Prediction")
    plt.axis("off")

    plt.show()
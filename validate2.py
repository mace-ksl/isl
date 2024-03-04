import torch
import model
import os 
import pytorch_lightning as pl 
import data_set
import matplotlib.pyplot as plt
import numpy as np

path = os.path.join(os.getcwd(),"data_set")
data = data_set.DataSet(path)
data_train_in = data.get_input_images_as_array("split_cy5","train")
data_train_out = data.get_output_images_as_array("split_cy5","train")

data_val_in = data.get_input_images_as_array("split_cy5","val")
data_val_out = data.get_output_images_as_array("split_cy5","val")

data_test_in = data.get_input_images_as_array("split_cy5","test")
data_test_out = data.get_output_images_as_array("split_cy5","test")

train_loader, val_loader, test_loader = data.create_torch_data_loader(data_train_in,data_train_out,data_val_in,data_val_out,data_test_in,data_test_out,height=256,width=256)


#model_train = model.Model("Unet", "mit_b2", in_channels=3, out_classes=1)
model_train = model.Model.load_from_checkpoint("E:/Data_sets/Github/timm/isl/data_set/model.ckpt", encoder_name="mit_b2" )
#model_train = model.Model.load_from_checkpoint("E:/Data_sets/Github/isl\data_set/lightning_logs/version_0/checkpoints/epoch=51-step=52.ckpt")
trainer = pl.Trainer(accelerator='gpu', devices=1,num_nodes=1, max_epochs=1, default_root_dir = data.data_dir)

#print(trainer.test(model_train,dataloaders=test_loader,verbose=False))

# Result visualization
batch = next(iter(test_loader))
with torch.no_grad():
    model_train.eval()
    logits = model_train(batch[0])
pr_masks = logits.sigmoid()

print(len(batch[0]))

for image, gt_mask, pr_mask in zip(batch[0], batch[1], pr_masks):
    plt.figure(figsize=(10, 5))

    print(image.shape)
    print(gt_mask.shape)
    print(pr_mask.shape)
    image = image.numpy()
    for i in range(3):
        plt.subplot(1, 6, i + 1)
        plt.imshow(image[i], cmap='gray')
        plt.title(f"Image {i+1}")
        plt.axis("off")

    #plt.subplot(1, 3, 1)
    #plt.imshow(image.numpy().transpose(1, 2, 0),cmap='gray')  # convert CHW -> HWC
    #plt.title("Image")
    #plt.axis("off")
    ground_truth_array = gt_mask.numpy()
    max_ground_truth = np.max(ground_truth_array)
    min_ground_truth = np.min(ground_truth_array)
    plt.subplot(1, 6, 4)
    plt.imshow(ground_truth_array.squeeze(),cmap='gray') # just squeeze classes dim, because we have only one class
    plt.title("Ground truth")
    plt.axis("off")
    print(ground_truth_array.shape)
    print(ground_truth_array)


    pr_mask_array = pr_mask.numpy()
    pr_mask_array = pr_mask_array * (max_ground_truth - min_ground_truth) + min_ground_truth
    print(pr_mask_array.shape)
    print(pr_mask_array)
    plt.subplot(1, 6, 5)
    plt.imshow(pr_mask_array.squeeze(),cmap='gray') # just squeeze classes dim, because we have only one class
    plt.title("Prediction")
    plt.axis("off")


    pr_mask[pr_mask < 0.52] = 0
    #print(type(pr_mask))
    min = torch.min(pr_mask)
    pr_mask = pr_mask - min
    # Plot with mask
    plt.subplot(1, 6, 6)
    plt.imshow(pr_mask.numpy().squeeze(),cmap='binary_r') # just squeeze classes dim, because we have only one class
    #print(pr_mask.numpy())
    plt.title("Prediction Mask")
    plt.axis("off")

    plt.show()
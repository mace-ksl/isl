import segmentation_models_pytorch as smp
import torch
import pytorch_lightning as pl 
import torchseg
import torch.nn as nn
import io
import numpy as np

class Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding="same")
        self.batch = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.batch(self.conv(x)))

class Model(pl.LightningModule):

    def __init__(self,model_path, encoder_name,learning_rate, **kwargs):
        super().__init__()

        # Create torchseg MaxViT model from timm
        # Paper: https://arxiv.org/abs/2204.01697 MaxViT: Multi-Axis Vision Transformer
        self.model = torchseg.Unet(
                    "maxvit_small_tf_224",
                    in_channels=3,
                    classes=1,
                    encoder_weights=True,
                    encoder_depth=5,
                    decoder_channels=(256, 128, 64, 32, 16),
                    encoder_params={"img_size": 256}
                )
        
        checkpoint = torch.load(model_path)
        cleared_state_dict = {key.replace('model.', '', 1) if key.startswith('model.') else key: value for key, value in checkpoint['state_dict'].items()}
        self.model.load_state_dict(cleared_state_dict, strict=False)

        # preprocessing parameteres
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        self.learning_rate = learning_rate
        #self.loss_fn_binary = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        #self.loss_fn = torch.nn.L1Loss()
        #self.loss_fn = torch.nn.CrossEntropyLoss()
        self.loss_fn_binary = torch.nn.BCEWithLogitsLoss()
        self.loss_fn = torch.nn.MSELoss()

        self.block_1 = Block(1, 2)
        self.block_2 = Block(2, 2)

    def forward(self, image):
        # normalize image
        image = (image - self.mean) / self.std
        mask = self.model(image)
        
        print("Image device:", image.shape)
        print("Mask device:", mask.shape)
        #Image device: torch.Size([1, 3, 256, 256])
        #Mask device: torch.Size([1, 1, 256, 256])
        # Upscaling
        output = self.block_1(mask)
        print(f"Block_1: {output.shape}")
        output = self.block_2(output)
        print(f"Block_2: {output.shape}")
        #Block_1: torch.Size([1, 2, 256, 256])
        #Block_2: torch.Size([1, 2, 256, 256])

        return output

    def shared_step(self, batch, stage):
        
        image = batch[0]
        #print(image.shape)
        #image = image.permute(0, 3, 1, 2)
        #print(image.shape)

        # Shape of the image should be (batch_size, num_channels, height, width)
        assert image.ndim == 4

        h, w = image.shape[2:]
        #print(h,w)
        assert h % 32 == 0 and w % 32 == 0

        mask = batch[1]
        
        #print(type(mask))

        mask = mask / torch.max(mask)

        #mask = mask / 255.0

        assert mask.ndim == 5

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        
        print(f"Mask shape {mask.shape} - Image shape {image.shape} - Logits shape {logits_mask.shape}")
        # Mask shape torch.Size([1, 1, 2, 256, 256]) - Image shape torch.Size([1, 3, 256, 256]) - Logits shape torch.Size([1, 2, 256, 256])
        mask=mask.squeeze(1)

  
        loss_l1 = self.loss_fn(logits_mask[:, 1:2, :, :], mask[:, 1:2, :, :])
        loss_fn_binary = self.loss_fn_binary(logits_mask[:, 0:1, :, :],mask[:, 0:1, :, :])

        #loss_l1 = self.loss_fn(logits_mask, mask)
        #loss_fn_binary = self.loss_fn_binary(logits_mask,mask)

        loss =  0.5*loss_l1 + 0.5*loss_fn_binary

        prob_mask = logits_mask.sigmoid()
        pred_mask = prob_mask
        #pred_mask = (prob_mask > 0.5).float()
   
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate) # lr=0.0001 
import segmentation_models_pytorch as smp
import torch
import pytorch_lightning as pl 
import torchseg
import torch.nn as nn
import io
import numpy as np

class Model(pl.LightningModule):

    def __init__(self, model_path,encoder_name,learning_rate, **kwargs):
        super().__init__()

        self.model = torchseg.Unet(
                    "maxvit_small_tf_224",
                    in_channels=3,
                    classes=1,
                    encoder_weights=True,
                    encoder_depth=5,
                    decoder_channels=(256, 128, 64, 32, 16),
                    encoder_params={"img_size": 256}
                )
        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))
        self.learning_rate = learning_rate
        
        # Load pretrained DAPI model
        checkpoint = torch.load(model_path)
        cleared_state_dict = {key.replace('model.', '', 1) if key.startswith('model.') else key: value for key, value in checkpoint['state_dict'].items()}
        self.model.load_state_dict(cleared_state_dict, strict=False)
        
        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        #self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
    
    def jaccard_loss(self,outputs, targets):
        eps=1e-7
        if outputs.dim() > 2:
            # Flatten the outputs and targets to ensure compatibility for broadcasting
            outputs = outputs.view(outputs.shape[0], -1)
            targets = targets.view(targets.shape[0], -1)

        # Calculate intersection and union
        intersection = (outputs * targets).sum(dim=1)  # Sum over all pixels
        total = outputs.sum(dim=1) + targets.sum(dim=1)  # Sum over all pixels
        union = total - intersection

        # Compute the Jaccard index
        J = (intersection + eps) / (union + eps)

        # Return the Jaccard loss
        return 1 - J.mean()  # Average over all batches
    
    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        
        #print("Image device:", image.device)
        #print("Mask device:", mask.device)
        return mask

    def shared_step(self, batch, stage):
        
        image = batch[0]
        #print(image.shape)
        #image = image.permute(0, 3, 1, 2)
        #print(image.shape)
        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        #print(h,w)
        assert h % 32 == 0 and w % 32 == 0

        mask = batch[1]
        
        #print(type(mask))
        #mask = mask.permute(0, 3, 1, 2)

        mask = mask / torch.max(mask)

        #mask = mask / 255.0
        print(image.shape,mask.shape)
        print(mask.max(),mask.min())
        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        #loss = self.loss_fn(logits_mask, mask)
        #loss = self.jaccard_loss(logits_mask, mask)
        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then
        # apply thresholding
        

        #loss = self.jaccard_loss(logits_mask, mask)
        loss = self.loss_fn(logits_mask, mask)

        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
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
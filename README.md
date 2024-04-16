## Cell and Nucleus Segmentation with In Silico Labeling using U-Net MaxViT
**ISL Project**

Training a U-Net MaxViT architecture for nucleus and cell segmentation, using in silico labeling.

## Prerequisites :wrench:
1. **The libraries used in this project can be installed with:**
   ```bash
   pip install -r requirements.txt
   ```
   and
   ```bash
   pip install -r requirements_torch.txt
   ```
2. Each of the folders: DAPI, CellMask, NucleusSegmentation and CellSegmentation contain:
   - train.py (to train a model)
   - validate.py (to validate and plot some trained model results)
   - config.yaml (to set hyperparameters like path to dataset, learning rate, ...)
3. The data_set folder is expected in the root folder of this project. It should include a tiff folder containing the dataset and a data_set_split folder
```
  isl/ 
  ├── .github/ 
  ├── data_set/
    └── tiff/ 
    └── data_set_split/
      └── split_dapi
        └── test.txt
        └── train.txt
        └── val.txt
```
- **.tiff/**: contains the .tiff and .png images of the training data
- **data_set_split/**: contain the different .txt files with the splitted data probes

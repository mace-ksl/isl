import os
from PIL import Image,ImageSequence
import numpy as np
import torch 
from torch.utils.data import DataLoader,TensorDataset
import plot
from skimage import io as skio

class DataSet():
    def __init__(self, data_dir):
        """
        Input:
            - data_dir: directory of the data set

        Vars:
            - self.data_dir: path of the data set
            - self.data_split_dir: /data_set_split folder containing the splitted data for training,text,val as .txt files
            - self.file_list: list of all files in data set folder
        """
        self.data_dir = data_dir
        self.data_split_dir = os.path.join(data_dir,"data_set_split")
        self.file_list = os.listdir(data_dir)
        print(f'Data set Object created successfully!\nData set directory is {self.data_dir}')
    
    def generate_input_image(self,file,mode=""):
        """
        Input:
            - mode: [bf,pc] brightfield or phase contrast
        """
        file_path = os.path.join(os.path.join(self.data_dir,"tiff256_nucleus_resized"),file)
        if mode == "bf":
            image = skio.imread(file_path)
            # brightfield image  H, W= H, W 
            return image
        elif mode == "pc":
            image = skio.imread(file_path)
            # phaste contrast image  H, W = H, W
            return image
        else:
            return None

    def generate_output_image(self,file):
        """
        Input:
            - self: Dataset object
            - file: File name of the .tif file
        Output:
            - output: Image as numpy array with shape(Channel=2,Height,Width)
        Method:
            Generate raw numpy images from .tiff files
            We want input image as: Channel, Height, Width = 3, H, W
            Grayscale 16 bit images
        """
        
        file_path = os.path.join(os.path.join(self.data_dir,"tiff256_nucleus_resized"),file)
        image = skio.imread(file_path)
        return image[:, :, np.newaxis]

    def get_input_images_as_array(self, split_dir, mode):
        """
        Input:
            - split_dir: name of the folder containing test.txt,train.txt,val.txt files with names of the data (e.g. split_dapi)
            - mode: mode of the data [test, train, val]
        Return:
            - Array of all data along the 0 axis: shape(image_count,channel,H,W) - (x,3,1024,1024)
        """
        
        file_path = os.path.join(self.data_split_dir,split_dir,mode+".txt")
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        stacked_arrays = []
        for line in lines:
            split = line.strip()
            image_path_brightfield = split+"_bf_ch_1.tiff"
            image_array_brightfield = self.generate_input_image(image_path_brightfield,mode="bf")

            image_path_brightfield2 = split+"_bf_ch_2.tiff"
            image_array_brightfield2 = self.generate_input_image(image_path_brightfield2,mode="bf")

            image_path_phase_contrast = split+"_pc.tiff"
            image_array_phast_contrast = self.generate_input_image(image_path_phase_contrast,mode="pc")
            
            combined_array = np.stack((image_array_brightfield, image_array_brightfield2, image_array_phast_contrast), axis=-1)
            # Transpose array from (H, W, C) to (C, H, W)
            transposed_array = np.transpose(combined_array, (2, 0, 1))
            stacked_arrays.append(transposed_array)
            
        return np.stack(stacked_arrays, axis=0)
    
    def get_output_images_as_array(self, split_dir, mode):
        """
        Input:
            - split_dir: name of the folder containing test.txt,train.txt,val.txt files with names of the data (e.g. split_cy5)
            - mode: mode of the data [test, train, val]
        Return:
            - Array of all data along the 0 axis: shape(image_count,channel,H,W) - (x,1,2160,2160)
        """
        file_path = os.path.join(self.data_split_dir,split_dir,mode+".txt")
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        stacked_arrays = []
        for line in lines:
            split = line.strip()
            image_path_dapi = split+"_cellpose.png"
            image_array_dapi = self.generate_output_image(image_path_dapi)
            # Transpose array from (H, W, C) to (C, H, W)

            transposed_array = np.transpose(image_array_dapi, (2, 0,1))
            stacked_arrays.append(transposed_array)
            
        return np.stack(stacked_arrays, axis=0)
    
    def create_torch_data_loader(self,x_train=None,y_train=None,x_val=None,y_val=None, x_test=None,y_test=None, batch_size=1,height=224,width=224):
        """
        Method:
            Create Tensor DataLoader and TensorDataset from uint16 numpy Array
        """
        #print(x_train)
        assert x_train is not None or y_train is not None
        assert x_test is not None or y_test is not None

        # Convert to int16 array if arrays are uint16
        if x_train.dtype == np.uint16:
            x_train = x_train.astype(np.int32)
        if y_train.dtype == np.uint16:
            y_train = y_train.astype(np.int32)
        if x_test.dtype == np.uint16:
            x_test = x_test.astype(np.int32)
        if y_test.dtype == np.uint16:
            y_test = y_test.astype(np.int32)
        
        train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train) )

        print(f"(CROPPED) - Number of images: {len(train_dataset)} - Dimension input image: {train_dataset[0][0].shape} - Dimension output image {train_dataset[0][1].shape}")
        #plot.plot_random_image_from_dataset(train_dataset)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True)
        
        test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
   
        #plot.plot_random_image_from_dataset(test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)

        # Create val data set only if its given 
        if x_val is not None and y_val is not None:
            if x_val.dtype == np.uint16:
                x_val = x_val.astype(np.int32)
            if y_val.dtype == np.uint16:
                y_val = y_val.astype(np.int32)
            val_dataset = TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val))
            #plot.plot_random_image_from_dataset(val_dataset)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)

            return train_loader,val_loader,test_loader
        
        return train_loader, test_loader
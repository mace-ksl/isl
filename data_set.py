import os
from PIL import Image,ImageSequence
import numpy as np
import torch 
from torch.utils.data import DataLoader,TensorDataset


class DataSet():
    def __init__(self, data_dir):
        """
        Input:
            - data_dir: directory of the data set
            
        self vars:
            - self.data_split_dir: /data_set_split folder containing the splitted data for training,text,val
            - self.file_list: list of all files in data set folder
        """
        # path = path of the .tiff files
        # E:\Data_sets\ISL_Cell_Seg\tiff
        self.data_dir = data_dir
        self.data_split_dir = os.path.join(data_dir,"data_set_split")
        self.file_list = os.listdir(data_dir)

    
    def generate_input_image(self,file):
        """
        File: Name of the .tif file
        Generate raw numpy images from .tiff files
        We want input image  C, H, W = 3, H, W
        First three .tiff images for 3 channel input 
        Color = 1 -> grayscale 16 bit
        """
        image = Image.open(os.path.join(os.path.join(self.data_dir,"tiff"),file))
        raw_image = []
        for i, frame in enumerate(ImageSequence.Iterator(image)):
            #print(f"Frame {i + 1} shape: {frame.size}")
            #print(f"Frame {i + 1} mode: {frame.mode}")
            raw_image.append(np.array(frame))
            if i == 2:
                break
                
        # input image  C, H, W = 3, H, W
        return np.array(raw_image)

    def generate_output_image(self,file):
        """
        File: Name of the .tiff file
        Generate raw numpy images from .tiff files
        We want output image C, H, W = 1, H, W
        First three .tiff images for 3 channel input 
        Color = 1 -> grayscale 16 bit
        """
        image = Image.open(os.path.join(os.path.join(self.data_dir,"tiff"),file))
        raw_image = []
        for i, frame in enumerate(ImageSequence.Iterator(image)):
            #print(f"Frame {i + 1} shape: {frame.size}")
            #print(f"Frame {i + 1} mode: {frame.mode}")
            raw_image.append(np.array(frame))
            # only the first images for one channel output
            break
        # return: input image  C, H, W = 3, H, W
        return np.array(raw_image)

        
    def get_input_images_as_array(self, split_dir, mode):
        """
        Input:
            - split_dir: name of the folder containing test.txt,train.txt,val.txt files with names of the data (e.g. split_cy5)
            - mode: mode of the data [test, train, val]
        Return:
            - Array of all data along the 0 axis: shape(image_count,channel,H,W) - (x,3,1024,1024)
        """
        
        file_path = os.path.join(self.data_split_dir,split_dir,mode+".txt")
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        stacked_arrays = []
        for line in lines:
            #split[0] = microscopy type (e.g. TWDIC3), split[1] = image id  (e.g. s54)
            split = line.strip().split()
            image_path = split[0]+"_w5DIC-oil-40x_"+split[1]+".TIF"
            image_array = self.generate_input_image(image_path)
            stacked_arrays.append(image_array)
            
        return np.stack(stacked_arrays, axis=0)
    
    def get_output_images_as_array(self, split_dir, mode):
        """
        Input:
            - split_dir: name of the folder containing test.txt,train.txt,val.txt files with names of the data (e.g. split_cy5)
            - mode: mode of the data [test, train, val]
        Return:
            - Array of all data along the 0 axis: shape(image_count,channel,H,W) - (x,1,1024,1024)
        """
        file_path = os.path.join(self.data_split_dir,split_dir,mode+".txt")
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        stacked_arrays = []
        for line in lines:
            #split[0] = microscopy type (e.g. TWDIC3), split[1] = image id  (e.g. s54)
            split = line.strip().split()
            image_path = split[0]+"_w4Cy5_"+split[1]+".TIF"
            image_array = self.generate_output_image(image_path)
            stacked_arrays.append(image_array)
            
        return np.stack(stacked_arrays, axis=0)
    
    def create_torch_data_loader(self,x_train,y_train,x_val,y_val, x_test,y_test, batch_size=10):
        """
        Method:
            Create Tensor DataLoader and TensorDataset from uint16 numpy Array
        """

        # Convert to int16 array if arrays are uint16
        if x_train.dtype == np.uint16:
            x_train = x_train.astype(np.int32)
        if y_train.dtype == np.uint16:
            y_train = y_train.astype(np.int32)
        if x_val.dtype == np.uint16:
            x_val = x_val.astype(np.int32)
        if y_val.dtype == np.uint16:
            y_val = y_val.astype(np.int32)
        if x_test.dtype == np.uint16:
            x_test = x_test.astype(np.int32)
        if y_test.dtype == np.uint16:
            y_test = y_test.astype(np.int32)
        

        #num_cpus = os.cpu_count()
        train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train) )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        val_dataset = TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



        return train_loader,val_loader,test_loader
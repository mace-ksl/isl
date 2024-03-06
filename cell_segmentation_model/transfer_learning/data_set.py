import os
from PIL import Image,ImageSequence
import numpy as np
import torch 
from torch.utils.data import DataLoader,TensorDataset
import augment
import plot

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
    
    def generate_input_image(self,file):
        """
        Input:
            - self: Dataset object
            - file: File name of the .tif file
        Output:
            - output: Image as numpy array with shape(Channel=3,Height,Width)
        Method:
            Generate a single raw numpy images from .tiff file
            We want input image as: Channel, Height, Width = 3, H, W
            Grayscale 16 bit images
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
            split = line.strip().split()
            #split[0] = microscopy type (e.g. TWDIC3)
            microscopy_type = split[0]
            #split[1] = image id  (e.g. s54)
            image_id = split[1]

            image_path_cell_topology = microscopy_type+"_cell_topology_"+image_id+".png"
            image_array_cell_topology = self.generate_output_image(image_path_cell_topology)

            image_path_cell_semantic = microscopy_type+"_cell_semantic_"+image_id+".png"
            image_array_cell_semantic = self.generate_output_image(image_path_cell_semantic)

            combined_output = np.vstack((image_array_cell_topology, image_array_cell_semantic))

            # Append output of shape (Channel=2,Height,Width)
            stacked_arrays.append(combined_output)
            
            
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
        train_dataset = augment.crop_dataset(train_dataset,height,width)
        #plot.plot_random_image_from_dataset(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True)
        
        test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
        test_dataset = augment.crop_dataset(test_dataset,height,width)
        #plot.plot_random_image_from_dataset(test_dataset)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)

        # Create val data set only if its given 
        if x_val != None and y_val != None:
            if x_val.dtype == np.uint16:
                x_val = x_val.astype(np.int32)
            if y_val.dtype == np.uint16:
                y_val = y_val.astype(np.int32)
            val_dataset = TensorDataset(torch.Tensor(x_val), torch.Tensor(y_val))
            val_dataset = augment.crop_dataset(val_dataset,height,width)
            #plot.plot_random_image_from_dataset(val_dataset)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)

            return train_loader,val_loader,test_loader
        
        return train_loader, test_loader
import pandas as pd
from PIL import Image
from torch.utils import data
from torchvision import transforms
import torch
import os

def getData(mode):
    if mode == 'train':
        curr_path = os.getcwd()
        path = os.path.join(curr_path, 'dataset/train.csv')
        print(path)
        df = pd.read_csv(path)
        path = df['filepaths'].tolist()
        label = df['label_id'].tolist()
        return path, label
    elif mode == 'valid':
        curr_path = os.getcwd()
        path = os.path.join(curr_path, 'dataset/valid.csv')
        print(path)
        df = pd.read_csv(path)
        path = df['filepaths'].tolist()
        label = df['label_id'].tolist()
        return path, label
    else:
        curr_path = os.getcwd()
        path = os.path.join(curr_path, 'dataset/test.csv')
        print(path)
        df = pd.read_csv(path)
        path = df['filepaths'].tolist()
        label = df['label_id'].tolist()
        return path, label

class BufferflyMothLoader(data.Dataset):
    def __init__(self, root, mode):
        """
        Args:
            mode : Indicate procedure status(training or testing)

            self.img_name (string list): String list that store all image names.
            self.label (int or float list): Numerical list that store all ground truth label values.
        """
        self.root = root
        self.img_name, self.label = getData(mode)
        self.mode = mode
        print("> Found %d images..." % (len(self.img_name)))  

    def __len__(self):
        """'return the size of dataset"""
        return len(self.img_name)

    def __getitem__(self, index):
        """something you should implement here"""

        """
           step1. Get the image path from 'self.img_name' and load it.
                  hint : path = root + self.img_name[index] + '.jpg'
           
           step2. Get the ground truth label from self.label
                     
           step3. Transform the .jpg rgb images during the training phase, such as resizing, random flipping, 
                  rotation, cropping, normalization etc. But at the beginning, I suggest you follow the hints. 
                       
                  In the testing phase, if you have a normalization process during the training phase, you only need 
                  to normalize the data. 
                  
                  hints : Convert the pixel value to [0, 1]
                          Transpose the image shape from [H, W, C] to [C, H, W]
                         
            step4. Return processed image and label
        """
        
        img_path = self.root + '/dataset/' + self.img_name[index]
        img = Image.open(img_path)
        
        # transform the data
        if self.mode == 'train':
            transform = transforms.Compose([
                transforms.RandomRotation(10),  # Randomly rotate the image by 10 degrees
                transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
                transforms.RandomVerticalFlip(),  # Randomly flip the image vertically
                # transforms.RandomResizedCrop(224),  # Randomly crop a portion of the image and resize it to 224x224
                # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Randomly adjust brightness, contrast, saturation, and hue
                # transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.5),  # Randomly apply Gaussian blur with a probability
                transforms.ToTensor(),
            ])

        else:
            transform = transforms.ToTensor()
        img = transform(img)
        
        label = torch.tensor(self.label[index], dtype=torch.long)

        return img, label


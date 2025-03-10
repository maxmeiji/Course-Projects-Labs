import os
from glob import glob
import torch
from torch import stack
from torch.utils.data import Dataset as torchData
import torchvision.transforms as transforms
from torchvision.datasets.folder import default_loader as imgloader
from torch import stack
import json
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image


def transform(mode):
    if mode == 'train':
        trans = transforms.Compose([
            transforms.Resize((64,64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.5,0.5,0.5],
                std = [0.5,0.5,0.5]
            )
        ])
    else:
        trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean = [0.5,0.5,0.5],
                std = [0.5,0.5,0.5]
            )
        ]) 

    return trans

def get_filenames(mode):
    # No validation/test graph in this lab
    if mode != 'train':
        return None
    else:
        with open(f'{mode}.json', 'r') as f:
            data = json.load(f)
        filename = list(data.keys())
    return filename   

def get_labels(mode):
    with open(f'{mode}.json', 'r') as f:
        data = json.load(f)
    with open('objects.json', 'r') as f:
        label_mapping = json.load(f)

    if mode == 'train':
        labels = list(data.values())
    else:
        labels = data
    # One-hot format (24)
    one_hot_list = []
    for idx in range(len(labels)):
        one_hot_temp = np.zeros(24, dtype=int)
        for element in range(len(labels[idx])):
            one_hot_temp[label_mapping[labels[idx][element]]] = 1
        one_hot_list.append(one_hot_temp)  # Keep as numpy array
    
    return np.array(one_hot_list)


class Dataset_iclevr(torchData):

    def __init__(self, root, mode='train'):
        super().__init__()
        assert mode in ['train', 'test', 'new_test'], "There is no such mode !!!"
        self.mode = mode
        self.transform = transform(mode)
        self.files = get_filenames(mode)
        self.labels = get_labels(mode)
        self.root = root

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # only training need to load the images
        img = torch.zeros((3, 64, 64))  # Default to a zero tensor with expected image shape
        if self.mode == 'train':
            file = self.files[index]
            img = Image.open(os.path.join(self.root, file)).convert('RGB')
            img = self.transform(img)
            
        label = self.labels[index]
        label = torch.Tensor(label)
        return img, label 


"""
if __name__ == '__main__':
    dataset = Dataset_iclevr(root='./iclevr', mode ='test')
    print(len(dataset))
    train_dataloader = DataLoader(
        Dataset_iclevr(root='./iclevr', mode='new_test'),
        batch_size=4,
        shuffle=True
    )
    for i, (img, label) in enumerate(train_dataloader):
        print(f"Batch {i}:")
        print(f"Image batch shape: {img.shape}")
        print(f"Label batch shape: {label.shape}")
        # Optionally break after first batch for demonstration
        break
        """








import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data.dataset import MyDataset, DataTransform
import os

def get_dataloaders(path='/media/ai/DATA/Hoang/emotions/dataset/fer2013', phases=['train','val','test'], target_size=96, batch_size=64):
    mean, std = 0, 255
    
    transform = DataTransform(mean, std, target_size)
    train_set = MyDataset(os.path.join(path, phases[0]), transform, phases[0])
    val_set = MyDataset(os.path.join(path, phases[1]), transform, phases[1])
    test_set = MyDataset(os.path.join(path, phases[2]), transform, phases[2])
    
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    return train_dataloader, val_dataloader, test_dataloader


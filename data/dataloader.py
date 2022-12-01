import os
import glob
from torch.utils.data import DataLoader
from data.dataset import MyDataset, DataTransform
from sklearn.model_selection import train_test_split
import shutil

def get_dataloaders(path="/media/ai/DATA/Hoang/emotions/dataset/CK+48", phases=['train','val','test'], 
                    target_size=96, batch_size=64, mean=0, std=255):
    
    transform = DataTransform(mean, std, target_size)
    train_set = MyDataset(os.path.join(path, phases[0]), transform, phases[0])
    val_set = MyDataset(os.path.join(path, phases[1]), transform, phases[1])
    test_set = MyDataset(os.path.join(path, phases[2]), transform, phases[2])
    
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_dataloader, val_dataloader, test_dataloader

if __name__ == "__main__":
    get_dataloaders()
    
    
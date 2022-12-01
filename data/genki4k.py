from torchvision import transforms
from torch.utils.data.dataset import Dataset
from PIL import Image
import cv2
import os
import numpy as np

class DataTransform(object):
    def __init__(self, mean, std, target_size):
        self.transform = {
            'train': transforms.Compose([
                    transforms.RandomResizedCrop(target_size, scale=(0.8, 1.2)),
                    transforms.RandomApply([transforms.RandomAffine(0, translate=(0.2, 0.2))], p=0.5),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomApply([transforms.RandomRotation(10)], p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=(mean,), std=(std,))
            ]),
            'val': transforms.Compose([
                transforms.Resize((target_size, target_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(mean,), std=(std,))
            ]),
            'test': transforms.Compose([
                transforms.Resize((target_size, target_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(mean,), std=(std,))
            ])
        }
    def __call__(self, img, phase='train'):
        return self.transform[phase](img)
    
class Genki4k(Dataset):
    def __init__(self, data_paths, ground_truths, transform=None, phase='train'):
        self.data_paths = data_paths
        self.ground_truths = ground_truths
        self.transform = transform
        self.phase = phase
    
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, index):
        img_path = self.data_paths[index]
        ground_truth = self.ground_truths[index]
        img = cv2.imread(img_path, 0)
        img = Image.fromarray(img)
        img_transformed = self.transform(img)
        return img_transformed, ground_truth
    
def make_data_paths(root_path='/Data/Hoang/emotions/data', labels='labels.txt'):
    
    data_paths = []
    ground_truths = []
    
    with open(os.path.join(root_path, labels), 'r') as f:
        lines = f.readlines()
    
    face_path = os.path.join(root_path, 'faces')
    list_face_paths = sorted(os.listdir(face_path))

    for i,file_name in enumerate(list_face_paths):
        data_paths.append(os.path.join(face_path, file_name))
        ground_truths.append(int(lines[i][0]))
        
    return np.array(data_paths), np.array(ground_truths)

if __name__ == "__main__":
    data_paths, ground_truths = make_data_paths()
    print(data_paths.shape)
    print(ground_truths.shape)
        
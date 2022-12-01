import torch
import cv2
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from PIL import Image
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def opencv_loader_gray(path):
    img = cv2.imread(path, 0)
    return img

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
                    # transforms.RandomErasing(p=0.5)
                    # transforms.TenCrop(48),
                    # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                    # transforms.Lambda(lambda tensors: torch.stack([transforms.Normalize(mean=(mean,), std=(std,))(t) for t in tensors])),
                    # transforms.Lambda(lambda tensors: torch.stack([transforms.RandomErasing(p=0.5)(t) for t in tensors])),
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
                # transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
                # transforms.Lambda(lambda tensors: torch.stack([transforms.Normalize(mean=(mean,), std=(std,))(t) for t in tensors]))
            ])
        }
    def __call__(self, img, phase='train'):
        return self.transform[phase](img)
    
class MyDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None, phase='train', loader=opencv_loader_gray):
        super(MyDataset, self).__init__(root)
        self.transform = transform
        self.phase = phase
        self.loader = loader
        # self.emotion_mapping = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
    
    def __getitem__(self, index):
        path, target = self.samples[index]
        
        try:
            img = self.loader(path) 
            img = Image.fromarray(img)
            img_transformed = self.transform(img, self.phase)
        except:
            print('path ', path)
            print('target ', target)
            pass
        # label = self.emotion_mapping[target]
        return img_transformed, target

def split_dataset(path='/home/hoangdinhhuy/Hoang/project_fgw/emotions/fer2013/test'):
    import glob
    import os
    from sklearn.model_selection import train_test_split
    import shutil
    class_names = os.listdir(path)
    class_list = []
    data_path_list = []
    for c in class_names:
        for img_path in glob.glob(os.path.join(path,c, "*.jpg")):
            data_path_list.append(img_path)
            class_list.append(c)
    # val_paths, test_paths, val_labels, test_labels = train_test_split(data_path_list, class_list, stratify=class_list, 
    #                                                                   test_size=0.5, random_state=42)
    # print(len(data_path_list), len(class_list))
    val_folder = path.replace('test', 'val')
    if not os.path.exists(val_folder):
        os.makedirs(val_folder)
    for p in data_path_list:
        name = os.path.basename(p)
        print('path ', p)
        if 'public' in name.lower():
            c = p.split('/')[-2]
            if not os.path.exists(os.path.join(val_folder,c)):
                os.makedirs(os.path.join(val_folder, c))
            shutil.move(p, os.path.join(val_folder,c, name))
            print('copied')
        print('==========================')

if __name__ == "__main__":
    train_root_path = '/home/hoangdinhhuy/Hoang/project_fgw/emotions/fer2013/train'
    mean, std = 0, 255
    target_size = 48
    transform = DataTransform(mean, std, target_size)
    train_set = MyDataset(train_root_path, transform, phase='train')
    img, label = train_set[0][0], train_set[0][1]
    img = img.numpy()
    cv2.imshow('a', img[0])
    # for i, im in enumerate(img):
    #     # print(img.shape)
    #     np_img = im[0].numpy()
    #     print(np_img.shape)
    #     cv2.imshow(f'n{i}', np_img)
    cv2.waitKey(0)
    
    # # Run only one time
    # split_dataset()
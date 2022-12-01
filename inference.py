import torch
import numpy as np
from data.dataset import DataTransform, MyDataset
import cv2
from PIL import Image
import torch
from tqdm import tqdm
from yaml import load
from torch.utils.data import DataLoader

from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt     
import pandas as pd
from models.efficientnet import Model
device = "cpu"
class_names = ['anger', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
print('device: ',device)
model_path = "/Data/Hoang/emotions/checkpoints/efficientnet_fer2013_64_AdamW_0.03/best.pth"
checkpoint = torch.load(model_path, "cpu")
model = Model(7)
model.load_state_dict(checkpoint['params'], strict=False)
# print(model.eval())
mean=0
std=255
target_size=256
data_transform = DataTransform(mean, std, target_size)
test_set = MyDataset("/Data/Hoang/emotions/dataset/org_fer2013/test", data_transform, "test")
test_dataloader = DataLoader(test_set, batch_size=64, shuffle=False)
outs = []
gts = []
with torch.no_grad():
    for (x, y) in tqdm(test_dataloader, desc='Testing', leave=False):
        x = x.to(device)
        y = y.to(device).numpy()
        y_pred = model(x)
        top_pred = list(y_pred.argmax(1, keepdim=True).squeeze().numpy())
        outs.extend(top_pred)
        gts.extend(y)
        
print(classification_report(gts, outs, target_names=class_names))

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
   
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        
    cm = confusion_matrix(y_true, y_pred)

    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    
    figure, axis = plt.subplots()
    im = axis.imshow(cm, interpolation='nearest', cmap=cmap)
    axis.figure.colorbar(im, ax=axis)

    axis.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),

           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='Actual',
           xlabel='Predict')

    # plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axis.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="cyan" if cm[i, j] > thresh else "red")
    figure.tight_layout()
    # plt.tight_layout()
    # plt.colorbar()
    # plt.show()
    # plt.savefig(cm)
    im.figure.savefig('fer2013_confusion_matrix.png')
    return axis

axis = plot_confusion_matrix(gts, outs, classes=class_names, normalize=False,
                      title='FER2013 Confusion Matrix')

# a = 
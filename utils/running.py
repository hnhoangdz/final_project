import warnings

import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
from utils.helper import calculate_accuracy
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from torch import autograd
warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(net, dataloader, criterion, optimizer, scaler):
    
    net = net.train()
    epoch_loss = 0.0
    epoch_acc = 0.0
    targets = []
    outputs = []
    # print('lllllll ', len(dataloader))
    for (x,y) in tqdm(dataloader, desc="Training", leave=False):
        # print("xxxxxxxxxxxxx ", x.shape)
        # print("yyyyyyyyyyyyy ", y.shape)
        targets.extend(y.cpu().int().numpy())
        x = x.to(device)
        y = y.to(device)
        
        #with autocast():
        # with torch.autograd.set_detect_anomaly(True):   
        optimizer.zero_grad()
        y_pred = net(x)
        
        top_pred, acc = calculate_accuracy(y_pred, y)
        outputs.extend(top_pred.cpu().int().numpy())
        
        loss = criterion(y_pred, y)
        scaler.scale(loss).backward()
        
        scaler.step(optimizer)
        scaler.update()
        # optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()

    f1 = f1_score(targets, outputs, average="macro")
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader) , f1

def evaluate(net, dataloader, criterion, desc="Evaluating"):
    
    net.eval()
    epoch_loss = 0
    epoch_acc = 0
    targets = []
    outputs = []
    
    with torch.no_grad():

        for (x, y) in tqdm(dataloader, desc=desc, leave=False):
            targets.extend(y.cpu().int().numpy())
            x = x.to(device)
            y = y.to(device)
            
            y_pred = net(x)
            loss = criterion(y_pred, y)

            top_pred, acc = calculate_accuracy(y_pred, y)
            outputs.extend(top_pred.cpu().int().numpy())

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            
    f1 = f1_score(targets, outputs, average="macro")
    
    if desc == "Testing":
        target_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        print(classification_report(targets, outputs, target_names=target_names))
        return 1
    
    return epoch_loss / len(dataloader), epoch_acc / len(dataloader), f1
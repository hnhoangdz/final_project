import sys
import warnings
import time
from tqdm import tqdm, trange
import optuna
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
import torch.optim as optim
from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import ReduceLROnPlateau

# from data.dataloader import get_dataloaders
from data.genki4k import DataTransform, Genki4k, make_data_paths
from utils.checkpoint import save
from utils.running import train, evaluate
from utils.setup_network import setup_network
from utils.helper import epoch_time, store_folder
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device: ", device)

if torch.cuda.is_available():
    torch.cuda.manual_seed(123)
    torch.cuda.synchronize() 
else:
    torch.manual_seed(123)
np.random.seed(123)

# CUDA_VISIBLE_DEVICES=2 python train_genki4k.py --bs 64 --target_size 48 --num_epochs 300 --lr 0.01 --optimizer AdamW --network resnet --data_name org_fer2013 --training_mode new > a.txt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True, default="fer2013")
    parser.add_argument("--bs", type=int, required=True, default=64,
                        help="Batch size of model")
    parser.add_argument("--target_size", type=int, required=True, default=48,
                        help="Image target size to resize")
    parser.add_argument("--num_epochs", type=int, required=True, default=300,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, required=True, default=0.1,
                        help="Learning rate")
    parser.add_argument("--training_mode", type=str, required=True, default="search",
                        help="Training model is defined to search params or not")
    parser.add_argument("--optimizer", type=str, required=True, default="SGD",
                        help="Optimizer to update weights")
    parser.add_argument("--network", type=str, required=True, default="cbam_resmob",
                        help="Name of network")
    parser.add_argument("--model_path", type=str, default="/home/ai/DATA/Hoang/emotions_v2/models",
                        help="Path to models folder that contains many models")
    parser.add_argument("--dataset_path", type=str, default="/home/ai/DATA/Hoang/emotions_v2/dataset")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--es_min_delta", type=float, default=0.0)
    parser.add_argument("--es_patience", type=int, default=5)
    parser.add_argument("--root_path", type=str,default="/home/ai/DATA/Hoang/emotions_v2/dataset/genki4k_data")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    args = vars(parser.parse_args())
    
    return args

def run(network, in_channels,
        batch_size, target_size, optimizer, 
        learning_rate, num_epochs, model_save_dir, 
        data_paths, ground_truths):
    
    target_size = 96
    mean = (0.5,)
    std = (0.5,)
    
    loss_logs = []
    acc_logs = []
    
    print('batch_size: ', batch_size)
    print('learning_rate: ', learning_rate)
    print('optimizer: ', optimizer)
    
    skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=123)
    # print("data_paths ", data_paths)
    for num_fold, (train_index, val_index) in enumerate(skf.split(data_paths, ground_truths)):
        
        # Init Dataset
        X_train_paths, X_val_paths = data_paths[train_index], data_paths[val_index]
        y_train_paths, y_val_paths = ground_truths[train_index], ground_truths[val_index]
        
        data_transform = DataTransform(mean, std, target_size)
        
        train_set = Genki4k(X_train_paths, y_train_paths, data_transform, phase="train")
        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
        
        val_set = Genki4k(X_val_paths, y_val_paths, data_transform, phase="val")
        val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
        
        # Init Model
        logger, net = setup_network(network, in_channels, 2)
        net = net.to(device)
        
        # Init Scaler
        scaler = GradScaler()
        
        # Init optimizer
        optimizer = AdamW(net.parameters(), lr= learning_rate)
        # if optimizer == 'SGD':
        #     optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, nesterov=True, weight_decay=0.001)

        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.75, patience=5, verbose=True)
        
        # Loss function
        criterion = nn.CrossEntropyLoss().to(device)       
        # store best val loss
        best_acc = 0.0
        best_val_acc = 0.0
        best_val_loss = 999.0
        
        for epoch in trange(num_epochs, desc="Epochs"):
            
            start_time = time.monotonic()
            loss_train, acc_train, f1_train = train(net, train_dataloader, criterion, optimizer, scaler)
            logger.loss_train.append(loss_train)
            logger.acc_train.append(acc_train)
            
            loss_val, acc_val, f1_val = evaluate(net, val_dataloader, criterion)
            logger.loss_val.append(loss_val)
            logger.acc_val.append(acc_val)

            scheduler.step(acc_val)
            
            end_time = time.monotonic()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            
            # save best checkpoint
            if acc_val > best_val_acc:
                best_val_acc = acc_val
                best_val_loss = loss_val
                save(net, logger, model_save_dir, "best")
                logger.save_plt(model_save_dir)
            
            # save last checkpoint (frequency)
            save(net, logger, model_save_dir, "last")
            logger.save_plt(model_save_dir)
                
            print(f'epoch: {epoch+1:02} | epoch time: {epoch_mins}m {epoch_secs}s')
            print(f'\ttrain loss: {loss_train:.3f} | train acc: {acc_train*100:.2f}% | train F1: {f1_train*100:.2f}%')
            print(f'\t val loss: {loss_val:.3f} |  val acc: {acc_val*100:.2f}% | val F1: {f1_val*100:.2f}%')
            
        loss_logs.append(best_val_loss)
        acc_logs.append(best_val_acc)
        
        print(f'STOP TRAINING FOLD: {num_fold}; best_val_acc: {best_val_acc}; best_val_loss: {best_val_loss}')
        
    for i in range(len(acc_logs)):
        print(f'> fold {i} ======> best val loss: {loss_logs[i]}; best val acc: {acc_logs[i]}')
    print('**Total Evaluation**')
    print(f'accuracy: {np.mean(acc_logs)}; loss: {np.mean(loss_logs)}; std acc: {np.std(acc_logs)}')
        
    return best_acc

def objective(trial):
    
    search_params = {
        'learning_rate': trial.suggest_categorical("learning_rate", [0.001, 0.01]),
        'optimizer': trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD", "AdamW"]),
    }
    
    print(sys.argv)
    print('[Loading] hyper-parameters..')
    model_save_dir = store_folder(args["data_name"], args["network"], args["bs"], search_params["optimizer"], 
                                  search_params["learning_rate"], available_nets)
    
    logger, net = setup_network(args["network"], in_channels)
    acc = run(net, logger, args["dataset_path"], args["data_name"], args["bs"], args["target_size"], search_params["optimizer"], 
                search_params["learning_rate"], args["num_epochs"], model_save_dir)
    return acc
  
if __name__ == "__main__":
    args = get_args()
    available_nets = set(filename.split(".")[0] for filename in os.listdir(args["model_path"]))
    available_datasets = set(filename for filename in os.listdir(args["dataset_path"]))
    in_channels = 1
    
    if args["data_name"] == "FERG":
        in_channels = 3

    model_save_dir = store_folder(args["data_name"], args["network"], args["bs"], args["optimizer"], args["lr"], available_nets)
    # logger, net = setup_network(args["network"], in_channels)
    data_paths, ground_truths = make_data_paths(args["root_path"])
    
    best_acc = run(args["network"], in_channels, args["bs"], args["target_size"], args["optimizer"], 
                    args["lr"], args["num_epochs"], model_save_dir, data_paths, ground_truths)
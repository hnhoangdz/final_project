import torch
import torch.nn as nn

def conv3x3(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, final=False):
        super(Block, self).__init__()
        self.final = final
        self.conv1 = conv3x3(in_channels, out_channels)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.maxp = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.final:
            out = self.conv2(out)
            
        out = self.maxp(out)
        return out

class Model(nn.Module):
    def __init__(self, in_channels=1, num_classes=7):
        super(Model, self).__init__()
        
        self.block1 = Block(in_channels, 32)
        self.block2 = Block(32, 64)
        self.block3 = Block(64, 128)
        self.block4 = Block(128, 256, final=True)
        self.flt = Flatten()
        
        self.fcn1 = nn.Linear(256, 256)
        self.fcn2 = nn.Linear(256, 256)
        
        self.bn = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, x):
        
        out = self.block1(x)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        
        out = self.flt(out)
        
        out = self.fcn1(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fcn2(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.classifier(out)
        
        return out
    
if __name__ == "__main__":
    import time
    import numpy as np
    model = Model(7)
    model.eval()
    print('total params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    x = torch.rand(1, 1, 96, 96)
    t = []
    for i in range(10):
        t0 = time.time()
        out = model(x)
        t.append(time.time() - t0)
    t = np.array(t)
    print('time inference: ', np.mean(t))
    print('delay time: ', np.std(t))    
        
        
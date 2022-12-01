import torch
import torch.nn as nn
from models.cbam import CBAM

def conv3x3(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels, affine=True, momentum=0.99, eps=1e-3),
        nn.ReLU()
    )

class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()
        
        self.conv = conv3x3(in_channels, out_channels, kernel_size=1, stride=2, padding=0)
        self.sep_conv1 = conv3x3(in_channels, out_channels, kernel_size=3, padding=1)
        self.sep_conv2 = conv3x3(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.cbam = CBAM(out_channels)
        self.maxp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        residual = self.conv(x)
        # residual = self.maxp(residual)
        out = self.sep_conv1(x)
        out = self.sep_conv2(out)
        out = self.cbam(out)
        out += residual 
        # out = self.relu(out)
        return out
    
class Model(nn.Module):
    def __init__(self, in_channels=1, num_classes=7):
        super(Model, self).__init__()
        self.relu = nn.ReLU()
        # 1
        self.conv1 = conv3x3(in_channels, 8)
        self.conv2 = conv3x3(8, 8)
        self.cbam = CBAM(8,8)
        
        # 2
        self.block1 = Block(8, 16)
        
        # 3
        self.block2 = Block(16, 32)
        
        # 4
        self.block3 = Block(32, 64)
        
        # 5
        self.block4 = Block(64, 128)
        
        # 
        self.last_conv = conv3x3(128, num_classes)
        
        # global avg-p
        self.avgp = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        
        # 1
        out = self.conv2(self.conv1(x))
        out = self.cbam(out)
        out = self.relu(out)
        # print('1', out.shape)
        # 2
        out = self.block1(out)
        # print('2', out.shape)
        # 3
        out = self.block2(out)
        # print('3', out.shape)
        # 4
        out = self.block3(out)
        # print('4', out.shape)
        # 5
        out = self.block4(out)
        # print('5555555555', out.shape)
        out = self.last_conv(out)
        out = self.avgp(out)
        out = out.view((out.shape[0], -1))
        return out   
    
if __name__ == "__main__":
    import time
    import numpy as np
    model = Model(1, 7)
    torch.save(model.state_dict(), "finalv2.pth")
    model.eval()
    print('total params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    x = torch.rand(1, 1, 48, 48)
    t = []
    for i in range(10):
        t0 = time.time()
        out = model(x)
        t.append(time.time() - t0)
    t = np.array(t)
    print('time inference: ', np.mean(t))
    print('delay time: ', np.std(t))
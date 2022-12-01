import torch.nn as nn
from models.cbam import CBAM
import torch

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class SeparableConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride=1, padding=padding, dilation=dilation, groups=in_channels,
                                   bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, stride, 0, 1, 1, bias=bias)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out
    
class ResidualBlock(nn.Module):
    
    def __init__(self, in_channel, out_channels):
        super(ResidualBlock, self).__init__()

        # normal conv -> down spatial dimension 2 times 
        self.residual_conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channels, kernel_size=3, stride=1, padding='same',
                                       bias=False)
        self.residual_bn = nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
        
        # down original spatial dimension 2 times
        self.maxp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) 

        # separable conv
        self.sepConv1 = SeparableConv2d(in_channels=in_channel, out_channels=out_channels, kernel_size=3, bias=False,padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)

        # separable conv
        self.sepConv2 = SeparableConv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=2, bias=False,padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=0.99, eps=1e-3)
        
        # convolutional block attention module
        self.cbam = CBAM(out_channels)

    def forward(self, x):
        # x = (b x c x h x w)
        residual = self.residual_conv(x) 
        residual = self.residual_bn(residual)
        residual = self.maxp(residual)

        out = self.sepConv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.sepConv2(out)
        out = self.bn2(out)
        
        out = self.cbam(out)
        out += residual
        out = self.relu(out)
        
        return out # (b x c x h//2 x w//2)

class Model(nn.Module):
    
    def __init__(self, in_channels=1, num_classes=7):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8,
                               kernel_size=3, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(8, affine=True, momentum=0.99, eps=1e-3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=8,
                               kernel_size=3, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(8, momentum=0.99, eps=1e-3)
        self.relu2 = nn.ReLU(inplace=True)

        self.module1 = ResidualBlock(in_channel=8, out_channels=16)
        self.module2 = ResidualBlock(in_channel=16, out_channels=32)
        self.module3 = ResidualBlock(in_channel=32, out_channels=64)
        self.module4 = ResidualBlock(in_channel=64, out_channels=128)
        self.fcn = nn.Sequential(
            Flatten(),
            nn.Linear(128*6*6, 256),
            nn.Dropout(0.4),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

        # self.last_conv = nn.Conv2d(in_channels=128, out_channels=num_classes, kernel_size=3, padding=1)
        # self.avgp = nn.AdaptiveAvgPool2d((1, 1))
            
    def forward(self, input):
        x = input
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.module4(x)
        print('xxxxxxxx', x.shape)
        x = self.fcn(x)
        # print('xxxxxxx: ', x.shape)
        return x

        x = self.last_conv(x)
        x = self.avgp(x)
        x = x.view((x.shape[0], -1))
        return x
    
if __name__ == "__main__":
    import time
    import numpy as np
    model = Model(7)
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
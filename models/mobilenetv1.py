import torch
import torch.nn as nn

class SeparableConv2d(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

def conv_3x3(in_channels, out_channels, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class Model(nn.Module):
    def __init__(self, in_channels=1, num_classes=7):
        super(Model, self).__init__()
                
        self.model = nn.Sequential(
            conv_3x3(in_channels, 32, 2),
            SeparableConv2d(32, 64, 1),
            SeparableConv2d(64, 128, stride=2),
            SeparableConv2d(128, 128, stride=1),
            SeparableConv2d(128, 256, stride=2),
            SeparableConv2d(256, 256, stride=1),
            SeparableConv2d(256, 512, stride=2),
            SeparableConv2d(512, 512, stride=1),
            SeparableConv2d(512, 512, stride=1),
            SeparableConv2d(512, 512, stride=1),
            SeparableConv2d(512, 512, stride=1),
            SeparableConv2d(512, 512, stride=1),
            SeparableConv2d(512, 1024, stride=2),
            SeparableConv2d(1024, 1024, stride=1),
            nn.AvgPool2d(2)
        )
        
        self.fc = nn.Linear(1024, num_classes)
        self.model.apply(self.init_weights)
        self.fc.apply(self.init_weights)
    
    def init_weights(self, layer):
        if type(layer) == nn.Conv2d:
            nn.init.kaiming_normal_(layer.weight, mode='fan_out')
        if type(layer) == nn.Linear:
            nn.init.normal_(layer.weight, std=1e-3)
        if type(layer) == nn.BatchNorm2d:
            nn.init.constant_(layer.weight, 1)
            nn.init.constant_(layer.bias, 0)
            
    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    import time
    import numpy as np
    model = Model(1, 7)
    print('total params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.eval()
    x = torch.rand(1, 1, 48, 48)
    t = []
    for i in range(10):
        t0 = time.time()
        out = model(x)
        t.append(time.time() - t0)
    t = np.array(t)
    print('time inference: ', np.mean(t))
    print('delay time: ', np.std(t))
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.cbam import CBAM

class Model(nn.Module):
    
    def __init__(self, in_channels, num_classes):
        super(Model, self).__init__()
        
        # calculate same padding:
        # (w - k + 2*p)/s + 1 = o
        # => p = (s(o-1) - w + k)/2
        
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                        out_channels=64,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        # (1(32-1)- 32 + 3)/2 = 1
                        padding=1), 
            nn.ReLU(),
            nn.Conv2d(in_channels=64,
                        out_channels=64,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                            stride=(2, 2)),
            CBAM(64, 16)
        )
        
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=64,
                        out_channels=128,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128,
                        out_channels=128,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                            stride=(2, 2)),
            CBAM(128, 16)
        )
        
        self.block_3 = nn.Sequential(        
            nn.Conv2d(in_channels=128,
                        out_channels=256,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256,
                        out_channels=256,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=1),
            nn.ReLU(),        
            nn.Conv2d(in_channels=256,
                        out_channels=256,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2),
                            stride=(2, 2)),
            CBAM(256, 16)
        )
        
          
        self.block_4 = nn.Sequential(   
            nn.Conv2d(in_channels=256,
                        out_channels=512,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=1),
            nn.ReLU(),        
            nn.Conv2d(in_channels=512,
                        out_channels=512,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=1),
            nn.ReLU(),        
            nn.Conv2d(in_channels=512,
                        out_channels=512,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=1),
            nn.ReLU(),            
            nn.MaxPool2d(kernel_size=(2, 2),
                            stride=(2, 2)),
            CBAM(512, 16)
        )
        
        self.block_5 = nn.Sequential(
            nn.Conv2d(in_channels=512,
                        out_channels=512,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=1),
            nn.ReLU(),            
            nn.Conv2d(in_channels=512,
                        out_channels=512,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=1),
            nn.ReLU(),            
            nn.Conv2d(in_channels=512,
                        out_channels=512,
                        kernel_size=(3, 3),
                        stride=(1, 1),
                        padding=1),
            nn.ReLU(),    
            nn.MaxPool2d(kernel_size=(2, 2),
                            stride=(2, 2)),
            CBAM(512, 16)             
        )
            
        self.classifier = nn.Sequential(
            nn.Linear(512*3*3, 4096),
            nn.ReLU(True),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Linear(4096, num_classes),
        )
            
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.detach().zero_()
                    
        #self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        
    def forward(self, x):

        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)

        x = x.view(x.size(0), -1)
        logits = self.classifier(x)

        return logits

if __name__ == "__main__":
    import time
    import numpy as np
    model = Model(in_channels=3, num_classes=7)
    torch.save(model.state_dict(), "vgg16_cbam.pth")
    model.eval()
    print('total params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    x = torch.rand(1, 3, 256, 256)
    t = []
    for i in range(10):
        t0 = time.time()
        out = model(x)
        t.append(time.time() - t0)
    t = np.array(t)
    print('time inference: ', np.mean(t))
    print('delay time: ', np.std(t))
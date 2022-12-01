from models.sam import SAM
from models.cam import CAM
import torch.nn as nn
import torch

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8, pool_types=['avg', 'max'], is_spatial=True):
        super(CBAM, self).__init__()
        self.cam = CAM(in_channels, reduction_ratio, pool_types)
        self.is_spatial = is_spatial
        if is_spatial:
            self.sam = SAM()
            
    def forward(self, x):
        out = self.cam(x)
        if self.is_spatial:
            out = self.sam(out)
        return out
    
if __name__ == "__main__":
    x = torch.rand(3, 4, 16, 16)
    a = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=1, stride=2,
                                       bias=False)
    b = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    print(b(x).shape)
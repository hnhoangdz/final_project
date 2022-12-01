import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models

class Model(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Model, self).__init__()
        self.model = models.googlenet()
        self.model.conv1.conv = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        
    def forward(self, x):
        # print(self.model)
        out = self.model(x)
        if len(out) >= 1:
            return out[1]
        return out
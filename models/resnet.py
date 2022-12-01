import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models

class Model(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Model, self).__init__()
        self.model = models.resnet34()
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.model.fc = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        
    def forward(self, x):
        out = self.model(x)
        return out

if __name__ == "__main__":
    import torch
    import time
    import numpy as np
    model = Model(1,7)
    torch.save(model.state_dict(), "resnet34.pth")
    print('total params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.eval()
    x = torch.rand(1, 1, 96, 96)
    t = []
    for i in range(10):
        t0 = time.time()
        out = model(x)
        t.append(time.time() - t0)
    t = np.array(t)
    print('time inference: ', np.mean(t))
    print('delay time: ', np.std(t))
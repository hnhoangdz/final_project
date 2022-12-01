import torch.nn as nn
import numpy as np
import math
import torchvision.models as models

class Model(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Model, self).__init__()
        self.model = models.MobileNetV2()
        self.model.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        
    def forward(self, x):
        out = self.model(x)
        return out
                
if __name__ == "__main__":
    import torch
    import time
    import numpy as np
    model = Model(1,7)
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
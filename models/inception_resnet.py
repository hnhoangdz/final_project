from facenet_pytorch import InceptionResnetV1
import torch.nn as nn
import torch

class Model(nn.Module):
    
    def __init__(self, pretrained_dataset='vggface2', in_channels=1, num_classes=2):
        super(Model, self).__init__()
        self.model = InceptionResnetV1(classify=True, pretrained=pretrained_dataset, 
                                       num_classes=num_classes)    
        self.model.conv2d_1a.conv = nn.Conv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    import time
    import numpy as np
    model = Model(num_classes=7)
    print(model)
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
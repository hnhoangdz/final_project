from turtle import forward
import torch
import torch.nn as nn
import torchvision.models as models
class Model(nn.Module):
    def __init__(self, num_classes=7):
        super(Model, self).__init__()
        self.model = models.efficientnet_b7(weights="EfficientNet_B7_Weights.DEFAULT")
        self.model.features[0][0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.classifier[1] = nn.Linear(2560, num_classes, bias=True)
    
    def forward(self, x):
        out = self.model(x)
        return out
# x = torch.rand(1, 1, 48, 48)
# model = models.efficientnet_b7(weights="EfficientNet_B7_Weights.DEFAULT")
# model.features[0][0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
# model.classifier[1] = nn.Linear(2560, 7, bias=True)
# print('total params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
# print(model(x).shape)
        
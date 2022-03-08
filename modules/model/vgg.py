import torch
from torch import nn
from torchvision import models


class VggModel(nn.Module):
    def __init__(self):
        super(VggModel, self).__init__()
        self.vgg19 = models.vgg19(pretrained=True)
        #self.vgg19.features[0] = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.vgg19.classifier[6] = torch.nn.Linear(in_features=4096, out_features=7, bias=True)

    def forward(self, x):
        return self.vgg19(x)

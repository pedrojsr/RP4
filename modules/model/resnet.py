import torch
import torchvision
from torch import nn


class ResnetModel(nn.Module):
    def __init__(self):
        super(ResnetModel, self).__init__()
        self.resnet18 = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_swsl')
        # self.resnet18 = torchvision.models.resnext50_32x4d(pretrained=True)
        # self.resnet18.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = torch.nn.Linear(num_ftrs, 7, bias = True)
        #self.resnet18.fc = torch.nn.Linear(in_features=512, out_features=7, bias=True)

    def forward(self, x):
        return self.resnet18(x)

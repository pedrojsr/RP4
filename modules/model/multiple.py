import torch
from torch import nn


# based on https://www.kaggle.com/hujunnn/pytorch-fer-2013

class MultipleModel(nn.Module):
    def __init__(self, vgg):
        super(MultipleModel, self).__init__()
        self.vggl = []
        for i in range(7):
            self.vggl.append( vgg[i] )
        
        #self.resnet = resnet
        self.fc = nn.Sequential(
            nn.Linear(in_features=7*7, out_features=7),
        )

    def forward(self, x):
        result = []
        for i in range(7):
            result.append( self.vggl[i](x) )
            #result_2 = self.resnet(x)

            result[i] = result[i].view(result[i].shape[0], -1)
        #result_2 = result_2.view(result_2.shape[0], -1)
        result = torch.cat(tuple(result), 1)

        y = self.fc(result)

        return y

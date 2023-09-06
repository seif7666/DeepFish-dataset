import torch.nn as nn
from .model import resnet18,resnet34

class PipelineModel(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.detector= resnet34(num_classes=num_classes,pretrained=False)

    def forward(self,x):
        print(self.training)
        if self.training:
            return self.detector((x['img'],x['annot'],x['number']))
        else:
            return self.detector(x)
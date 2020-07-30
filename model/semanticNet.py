from SemanticNet.PretrainedBackbone import TwoWayFPNBackbone
from SemanticNet.semantichead import SemanticNet
import torch.nn as nn


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.nc = 20
        self.backbone = TwoWayFPNBackbone()
        self.semantic = SemanticNet()

        self.conv = nn.Conv2d(512, self.nc, 1)

    
    def forward(self, x):

        x = self.backbone(x)
        x = self.semantic(x['0'], x['1'], x['2'], x['3'])

        x = self.conv(x)
        
        return cat

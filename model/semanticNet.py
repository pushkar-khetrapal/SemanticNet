from SemanticNet.model.PretrainedBackbone import TwoWayFPNBackbone
from SemanticNet.model.semantichead import SemanticHead
import torch.nn as nn


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.backbone = TwoWayFPNBackbone().to('cuda')
        self.semantic = SemanticHead().to('cuda')
    
    def forward(self, x):

        x = self.backbone(x)
        x = self.semantic(x['3'], x['2'], x['1'], x['0'])
        
        return x

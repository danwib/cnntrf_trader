import torch.nn as nn
class CNNTransformer(nn.Module):
    def __init__(self,in_channels:int=8):
        super().__init__()
        self.net=nn.Linear(in_channels,1)
    def forward(self,x):
        return self.net(x.mean(-1))

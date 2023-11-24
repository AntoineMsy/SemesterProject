import copy
import torch.nn as nn

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Transpose(nn.Module):
    def __init__(self):
        super(Transpose, self).__init__()
    def forward(self,x):
        return x.transpose(1,2)
    
import torch
import torch.nn as nn
from models.model_utils import Transpose

class LinearEmbedding(nn.Module):
    def __init__(self, in_features, d_model):
        super(LinearEmbedding,self).__init__()
        self.lin = nn.Linear(in_features, d_model, bias=False)

    def forward(self, x):
        return self.lin(x)
    
class MLPEmbedding(nn.Module):
    def __init__(self, in_features, d_model, num_layers = 2):
        super(MLPEmbedding,self).__init__()
        # N-layers (standard 2) MLP  point embedding
        
        self.lbr_list = [nn.Linear(in_features, d_model, bias=False), Transpose(), nn.BatchNorm1d(d_model), Transpose(), nn.ReLU()]
        
        for i in range(num_layers-1):
            self.lbr_list += [nn.Linear(d_model, d_model, bias=False), Transpose(), nn.BatchNorm1d(d_model), Transpose(), nn.ReLU()]

        self.model = nn.Sequential(*self.lbr_list)

    def forward(self, x):
        return self.model(x)
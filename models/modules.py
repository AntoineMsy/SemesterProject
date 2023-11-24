import torch
import torch.nn as nn
import torch.nn.functional as F
from models.model_utils import clones, Transpose

class LayerNorm(nn.Module):
    # Construct a layernorm module (See citation for details).

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
    
class TransformerEncoder(nn.Module):
    def __init__(self,
                 layer,
                 num_layers,):
        super(TransformerEncoder,self).__init__()
        self.layers = clones(layer, num_layers)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, mask):
        # Pass the input (and mask) through each layer in turn.
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class PositionwiseFeedForward(nn.Module):
    # Implements FFN equation.

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class MLP_Decoder(nn.Module):
    def __init__(self,d_enc: int = 1024, d_int: int = 256, out_features: int = 3):
        #Implements a 3 layer MLP decoder
        super(MLP_Decoder, self).__init__()
        self.dropout = nn.Dropout()
        transpose = Transpose()
        bn = nn.BatchNorm1d(d_int)
        self.bn_t = nn.Sequential(transpose, bn, transpose)
        self.dec = nn.Sequential(nn.Linear(d_enc,d_int), self.dropout, self.bn_t, nn.ReLU(),
                                 nn.Linear(d_int,d_int), self.bn_t, nn.ReLU(),
                                 nn.Linear(d_int,out_features))
        
    def forward(self,x):
        return self.dec(x)
    
class SALayer(nn.Module):
    # Implements a MultiHead Self Attention layer
    def __init__(self, 
                 d_model, 
                 d_ff, 
                 num_heads,dropout = 0.1):
        super(SALayer,self).__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, batch_first= True)
        self.d_model = d_model
        self.ffn = PositionwiseFeedForward(d_model,d_ff)
        self.norm = LayerNorm(d_model)
        # self.lin_q = nn.Linear(d_model,d_model)
        # self.lin_k = nn.Linear(d_model,d_model)
        # self.lin_v = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask):
        x_n = self.norm(x)
        x_att = x + self.dropout(self.mha(x_n, x_n, x_n, mask, need_weights = False)[0])
        y = x_att + self.dropout(self.ffn(self.norm(x_att)))
        return y
    
    def get_weights(self,x,mask):
        x_n = self.norm(x)
        y = self.mha(x_n, x_n, x_n, mask)[1]
        return y

class OALayer(nn.Module):
    # TODO Implements an Offset Attention layer (see PCT paper) 
    def __init__(self, 
                 d_model, 
                 d_ff, 
                 num_heads,dropout = 0.1):
        super(SALayer,self).__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, batch_first= True)
        self.d_model = d_model
        self.ffn = PositionwiseFeedForward(d_model,d_ff)
        self.norm = LayerNorm(d_model)
        # self.lin_q = nn.Linear(d_model,d_model)
        # self.lin_k = nn.Linear(d_model,d_model)
        # self.lin_v = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask):
        x_n = self.norm(x)
        x_att = x + self.dropout(self.mha(x_n, x_n, x_n, mask, need_weights = False)[0])
        y = x_att + self.dropout(self.ffn(self.norm(x_att)))
        return y
    
    def get_weights(self,x,mask):
        x_n = self.norm(x)
        y = self.mha(x_n, x_n, x_n, mask)[1]
        return y



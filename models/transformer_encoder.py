import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

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
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))

class EncoderLayer(nn.Module):
    def __init__(self, 
                 d_model, 
                 d_ff, 
                 num_heads,dropout = 0.1):
        super(EncoderLayer,self).__init__()
        self.mha = nn.MultiheadAttention(d_model, num_heads, batch_first= True)
        self.d_model = d_model
        self.ffn = PositionwiseFeedForward(d_model,d_ff)
        self.norm = LayerNorm(d_model)
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

class TransformerSeg(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, num_layers, in_features, out_features):
        super(TransformerSeg,self).__init__()
        enc_layer = EncoderLayer(d_model,d_ff,num_heads, dropout = 0.01)
        self.encoder = TransformerEncoder(enc_layer,num_layers)
        self.lin_emb = nn.Linear(in_features,d_model)
        self.out_lin = nn.Linear(d_model,out_features)

    def forward(self,x, mask):
        x_model = torch.stack([self.lin_emb(x[:,i,:]) for i in range(x.shape[1])], dim=1)
        x_model_out = self.encoder(x_model, mask)
        y_pred = torch.stack([self.out_lin(x_model_out[:,i,:]) for i in range(x_model_out.shape[1])], dim=1)
        return y_pred

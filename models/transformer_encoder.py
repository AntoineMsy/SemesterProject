import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules import *
from models.embeddings import *

class TransformerEncoder(nn.Module):
    def __init__(self,
                 layer,
                 num_layers,norm_dim):
        super(TransformerEncoder,self).__init__()
        self.layers = clones(layer, num_layers)
        self.norm = LayerNorm(norm_dim)
        # self.norm_agg = LayerNorm(layer.d_model*num_layers)

    def forward(self, x, mask):
        # Pass the input (and mask) through each layer in turn.
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

    def forward_agg(self, x, mask):
        # Pass the input (and mask) through each layer in turn.
        out_l = []
        for layer in self.layers:
            x = layer(x, mask)
            out_l.append(x)
        return self.norm(torch.cat(out_l,dim=-1))
    
class TransformerSeg_pre(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, num_layers, in_features, out_features):
        super(TransformerSeg_pre,self).__init__()
        enc_layer = SALayer(d_model,d_ff,num_heads, dropout = 0.01)
        self.encoder = TransformerEncoder(enc_layer,num_layers, d_model)
        self.lin_emb = nn.Linear(in_features,d_model)
        self.out_lin = nn.Linear(d_model,out_features)

    def forward(self,x, mask):
        x_model = torch.stack([self.lin_emb(x[:,i,:]) for i in range(x.shape[1])], dim=1)
        x_model_out = self.encoder(x_model, mask)
        y_pred = torch.stack([self.out_lin(x_model_out[:,i,:]) for i in range(x_model_out.shape[1])], dim=1)
        return y_pred
    
class TransformerSeg(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, num_layers, in_features, out_features):
        super(TransformerSeg,self).__init__()
        enc_layer = SALayer(d_model,d_ff,num_heads, dropout = 0.01)
        self.encoder = TransformerEncoder(enc_layer,num_layers, d_model)
        self.lin_emb = nn.Linear(in_features,d_model, bias=False)
        self.out_lin = nn.Linear(d_model,out_features)

    def forward(self,x, mask):
        x_model = self.lin_emb(x)
        x_model_out = self.encoder(x_model, mask)
        y_pred = self.out_lin(x_model_out) 
        return y_pred
    
class TransformerSeg_v0(nn.Module):
    # Baseline transformer version. Can change the embeddings from simple linear layers to N-layers MLP or (TODO) MLP with local aggregation
    def __init__(self, d_model: int = 128, d_ff: int = 64, num_heads: int = 1, num_layers: int = 4, in_features: int = 4, out_features: int  = 3, emb_type: str = "lin"):
        super(TransformerSeg_v0,self).__init__()
        enc_layer = SALayer(d_model,d_ff,num_heads, dropout = 0.01)
        self.encoder = TransformerEncoder(enc_layer,num_layers, d_model)
        if emb_type == "lin":
            self.emb = LinearEmbedding(in_features,d_model)
        elif emb_type == "mlp":
            self.emb = MLPEmbedding(in_features,d_model, num_layers=2)
        else : 
            raise RuntimeError("Unrecognized embedding type")
   
        self.out_dec = MLP_Decoder(d_enc = d_model, out_features=out_features)
    def forward(self,x, mask):
        # x_model = torch.stack([self.emb(x[:,i,:]) for i in range(x.shape[1])], dim=1)
        x_model = self.emb(x)
        x_model_out = self.encoder(x_model, mask)
        y_pred = self.out_dec(x_model_out)
        return y_pred

class TransformerSeg_v1(nn.Module):
    #First test improvement over the Baseline. 
    # Implements the same features as the previous one, but also ideas from Point Cloud Transformer (https://arxiv.org/abs/2012.09688v4) :
    #   - Feature aggregation across all attention layers for final point prediction with an MLP
    #   - Offset Attention, which approximately multiplies the input of the layer with the Laplacian matrix of the fully connected graph of the input with attention weights as edge weights.

    def __init__(self, d_model: int = 128, d_ff: int = 64, num_heads: int = 1, num_layers: int = 4, in_features: int = 4, out_features: int  = 3):
        super(TransformerSeg_v1,self).__init__()
        enc_layer = SALayer(d_model,d_ff,num_heads, dropout = 0.01)
        self.encoder = TransformerEncoder(enc_layer,num_layers, d_model*num_layers)
        self.emb = MLPEmbedding(in_features,d_model, num_layers=2)
        self.out_dec = MLP_Decoder(d_enc = d_model*num_layers, out_features=out_features)

    def forward(self,x, mask):
        x_model = self.emb(x)
        x_model_out = self.encoder.forward_agg(x_model, mask)
        y_pred = self.out_dec(x_model_out)
        return y_pred

class TransformerSeg_v2(nn.Module):
    def __init__(self, d_model, d_ff, num_heads, num_layers, in_features, out_features):
        super(TransformerSeg_v2,self).__init__()
        enc_layer = SALayer(d_model,d_ff,num_heads, dropout = 0.01)
        self.encoder = TransformerEncoder(enc_layer,num_layers)
        self.lin_emb = nn.Linear(in_features,d_model)
        self.out_lin = nn.Linear(d_model,out_features)

    def forward(self,x, mask):
        x_model = torch.stack([self.lin_emb(x[:,i,:]) for i in range(x.shape[1])], dim=1)
        x_model_out = self.encoder(x_model, mask)
        y_pred = torch.stack([self.out_lin(x_model_out[:,i,:]) for i in range(x_model_out.shape[1])], dim=1)
        return y_pred
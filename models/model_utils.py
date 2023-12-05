import copy
import torch.nn as nn
import torch

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Transpose(nn.Module):
    def __init__(self):
        super(Transpose, self).__init__()
    def forward(self,x):
        return x.transpose(1,2)

def get_batch_from_data(dataset, idx):
    elem = dataset[idx]
    coords, label = elem["coords"], elem["values"]
    mask = torch.ones(len(coords))
    return coords[None,:], label[None,:], mask[None,:]

def model_inference(model, dataset, idx):
    coords, label, mask = get_batch_from_data(dataset,idx)
    return (torch.argmax(model(coords,mask), dim=2) +1).view(-1,1)

def get_attention_from_data(model,dataset,idx):
    coords, label, mask = get_batch_from_data(dataset,idx)
    coords = torch.stack([model.model.lin_emb(coords[:,i,:]) for i in range(coords.shape[1])], dim=1)
    out_first_layer = model.model.encoder.layers[0].get_weights(coords,mask)
    return out_first_layer
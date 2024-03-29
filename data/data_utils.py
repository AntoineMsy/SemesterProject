import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_sequence
import yaml
from scipy.spatial.transform import Rotation as R

"""
Basic data processing functions
"""

def charge_transform(x, mean = 185, log_std = 4):
    return np.log(x/mean)/log_std

def inv_charge_transform(y, mean = 185, log_std = 4):
    return mean*np.exp(log_std*y)

def scale_coords(x, max_mean = 540):
    return x/max_mean

def inv_scale_coords(y, max_mean = 540):
    return y*max_mean

def parse_yaml(file_path):
    with open(file_path, "r") as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            return None


class FLoss(nn.Module):
    """Implementation of the Focal Loss"""
    def __init__(self, weight = torch.tensor([1,1,1]), alpha=1, gamma=2):
        super(FLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=weight,reduction = "none")

    def forward(self, input, target,mask):
        ce_loss = self.ce(input,target,mask)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return torch.mean(focal_loss)


class Small_random_rot(object):
    """Data Augmentation with small random rotation on the input"""
    def __init__(self):
        self.p = np.pi
    def __call__(self, x):
        r_angles = [np.random.uniform(-self.p/8,self.p/8) for i in range(3)]
        r_mat = torch.from_numpy(R.from_euler("XYZ", r_angles).as_matrix())
        return torch.matmul(r_mat,x)

"""
Custom collate functions and sampler for optimized data loading
"""  

def pad_elem(el, mask_l, max_l):
    mask_seq = torch.zeros(max_l)
    mask_seq[:mask_l] = 1
    return [torch.nn.functional.pad(el, (0,0,0, max_l - mask_l), value = 0)[None,:], mask_seq]

def my_collate(batch_list):
    """Makes mask according to eahc sequence length and return a batch"""
    coords = [item["coords"] for item in batch_list]
    target = [item["values"] for item in batch_list]
    mask_lengths = [len(item["coords"]) for item in batch_list]
    max_l = max(mask_lengths)
    mask = torch.zeros(len(batch_list), max_l)
    feats_out = torch.stack([torch.nn.functional.pad(coords[i], (0,0,0, max_l - mask_lengths[i]), value = 0) for i in range(len(coords))])
    vals_out = torch.stack([torch.nn.functional.pad(target[i], (0,0,0, max_l - mask_lengths[i]), value = 0) for i in range(len(coords))])

    for i in range(len(batch_list)):
        mask[i,:mask_lengths[i]] = 1
        
    return {"coords": feats_out, "values": vals_out, "mask": mask }

class LenMatchBatchSampler(torch.utils.data.BatchSampler):
    """
    Custom Sampler that yields batches of approximately equal number of hits 
    by iterating over the dataset, putting indices into bucket of length (modulo 32) and yielding the bucket when its size equates the batch size
    """
    def __init__(self, data_source, sampler, batch_size, drop_last):
        super().__init__(sampler, batch_size, drop_last)
        self.dataset = data_source
    def __iter__(self):
        buckets = [[]] * 150
        yielded = 0
        
        for idx in self.sampler:
            #s = self.sampler.data_source[idx]
            s = self.dataset[idx]
            L = s["coords"].shape[0]
            
            # if isinstance(s, tuple):
            #     L = s[0]["mask"].sum()
            # else:
            #     L = s["mask"].sum()
            # if torch.rand(1).item() < 0.1: L = int(1.5*L)
            L = max(0, L // 32)
            if len(buckets[L]) == 0:
                buckets[L] = []
            buckets[L].append(idx)

            if len(buckets[L]) == self.batch_size:
                batch = list(buckets[L])
                yield batch
                yielded += 1
                buckets[L] = []

        batch = []
        leftover = [idx for bucket in buckets for idx in bucket]
        print("loop finished, entering leftover indices")
        # I tried reducing the batch size during the end process where all indices are yielded together (and thus big lengths comes at play and the memory usage explodes)
        # self.batch_size = self.batch_size//16
        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yielded += 1
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch
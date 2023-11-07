import torch
import numpy as np
from torch.nn.utils.rnn import pack_sequence

def charge_transform(x, mean = 185, log_std = 4):
    return np.log(x/mean)/log_std

def inv_charge_transform(y, mean = 185, log_std = 4):
    return mean*np.exp(log_std*y)

def scale_coords(x, max_mean = 540):
    return x/max_mean

def inv_scale_coords(y, max_mean = 540):
    return y*max_mean

def my_collate(batch_list):
    # batch contains a list of tuples of structure (sequence, target)
    # coords, target, mask = batch["coords"], batch["values"], batch["mask"]
    coords = [item["coords"] for item in batch_list]
    target = [item["values"] for item in batch_list]
    mask_lengths = [len(item["coords"]) for item in batch_list]
    max_l = max(mask_lengths)
    mask = torch.zeros(len(batch_list), max_l)

    feats_out = torch.empty(0)
    vals_out = torch.empty(0)
    for i in range(len(batch_list)):
        p2d = (0,0,0, max_l - mask_lengths[i])
        mask[i,:mask_lengths[i]] = 1
        feats_out, vals_out = torch.concatenate((feats_out,torch.nn.functional.pad(coords[i], p2d, value = 0)[None,:])), torch.concatenate((vals_out,torch.nn.functional.pad(target[i], p2d, value = 0)[None,:]))
    
    return {"coords": feats_out, "values": vals_out, "mask": mask }
    # coords = pack_sequence(coords, enforce_sorted=False)
    
    # target = pack_sequence(target, enforce_sorted=False)
    # return [coords, target]

class LenMatchBatchSampler(torch.utils.data.BatchSampler):
    def __init__(self, data_source, sampler, batch_size, drop_last):
        super().__init__(sampler, batch_size, drop_last)
        self.dataset = data_source
    def __iter__(self):
        buckets = [[]] * 5000
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
        print("loop finished")
        print(leftover)
        for idx in leftover:
            batch.append(idx)
            if len(batch) == self.batch_size:
                print("end yield")
                print(batch)
                yielded += 1
                yield batch
                batch = []

        if len(batch) > 0 and not self.drop_last:
            yielded += 1
            yield batch
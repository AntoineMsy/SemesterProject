import torch
import numpy as np
from torch.nn.utils.rnn import pack_sequence
import yaml
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
        
def pad_elem(el, mask_l, max_l):
    mask_seq = torch.zeros(max_l)
    mask_seq[:mask_l] = 1
    return [torch.nn.functional.pad(el, (0,0,0, max_l - mask_l), value = 0)[None,:], mask_seq]

def my_collate(batch_list):
    # coords, target, mask = batch["coords"], batch["values"], batch["mask"]
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
        # print(buckets)
        leftover = [idx for bucket in buckets for idx in bucket]
        print("loop finished, entering leftover indices")
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
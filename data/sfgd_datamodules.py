import torch
import os
import numpy as np
from torch.utils.data import Dataset, RandomSampler, random_split
import lightning.pytorch as pl
from data_utils import LenMatchBatchSampler
from torchvision import transforms

class NodeCL_dataset(Dataset):
    def __init__(self, data_dir):
        super(NodeCL_dataset, self).__init__()
        self.data_dir = data_dir
        self.len = len(os.listdir(self.data_path))

    def __getitem__(self,x):
        npz_file = np.load(self.data_path + "event%d.npz"%(x))
        n_hits = len(npz_file["c"])
        p2d = (0,0,0,64 - n_hits%64)
        mask = torch.zeros(n_hits+ 64 - n_hits%64)
        mask[:n_hits] = 1
        t_coords, t_vals = torch.nn.functional.pad(torch.tensor(npz_file["c"]-npz_file["verPos"]), p2d, value = 0), torch.nn.functional.pad(torch.tensor(npz_file["y"]), p2d, value = 0)
        return {"coords": t_coords, "values": t_vals, "mask": mask }
    
    def __len__(self):
        return self.len
    
class SFGD_tagging(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size):
        super(SFGD_tagging, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transforms = transforms.Compose([transforms.ToTensor()])
        
    def setup(self, stage):
        dataset = NodeCL_dataset(self.data_dir)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, lengths=[0.65,0.15,0.2])
    
    def train_dataloader(self):
        rand_sampler = RandomSampler(self.train_dataset)
        lenmatch_sampler = LenMatchBatchSampler(self.train_dataset, rand_sampler, batch_size= self.batch_size, drop_last=False)
  
    def val_dataloader(self):
        rand_sampler = RandomSampler(self.val_dataset)
        lenmatch_sampler = LenMatchBatchSampler(self.val_dataset, rand_sampler, batch_size= self.batch_size, drop_last=False)
  
    def test_dataloader(self):
        rand_sampler = RandomSampler(self.test_dataset)
        lenmatch_sampler = LenMatchBatchSampler(self.test_dataset, rand_sampler, batch_size= self.batch_size, drop_last=False)
  
    def teardown(self, stage):
        ...

import torch
import os
import numpy as np
import h5py
from torch.utils.data import Dataset, RandomSampler, random_split, DataLoader
import lightning.pytorch as pl
from data.data_utils import LenMatchBatchSampler, charge_transform, scale_coords, my_collate
from torchvision import transforms
from sklearn.preprocessing import OneHotEncoder

class NodeCL_dataset(Dataset):
    def __init__(self, data_dir):
        super(NodeCL_dataset, self).__init__()
        self.data_dir = data_dir
        self.len = len(os.listdir(self.data_dir))
        # self.transforms = transforms.Compose([transforms.ToTensor()])
        self.enc = OneHotEncoder(sparse_output=False)
        fit_array = np.array([[1],[2],[3]])
        self.enc.fit(fit_array)

    def __getitem__(self,x):
        npz_file = np.load(self.data_dir + "event%d.npz"%(x))
        n_hits = len(npz_file["c"])
        charge = charge_transform(npz_file["x"][:,1][:,None].astype("float32"))
        coords = npz_file["c"].astype("float32")-npz_file["verPos"].astype("float32")
        coords = scale_coords(coords)
        vals = self.enc.transform(npz_file["y"].astype("float32"))
        feats = np.concatenate([coords,charge], axis=1)
        # p2d = (0,0,0,64 - n_hits%64)
        # mask = torch.zeros(n_hits+ 64 - n_hits%64)
        # mask[:n_hits] = 1
        # t_coords, t_vals = torch.nn.functional.pad(torch.tensor(feats), "p2d", value = 0), torch.nn.functional.pad(torch.tensor(vals), "p2d", value = 0)
        t_coords, t_vals = torch.tensor(feats), torch.tensor(vals)
       
        return {"coords": t_coords, "values": t_vals}
    
    def __len__(self):
        return self.len

class NodeCL_h5dataset(Dataset):
    def __init__(self, data_dir):
        super(NodeCL_h5dataset, self).__init__()
        self.data_dir = data_dir
      
        self.h5_file = h5py.File(self.data_dir)
        self.len = len(self.h5_file["event_hits_index"])
        # self.transforms = transforms.Compose([transforms.ToTensor()])
        self.enc = OneHotEncoder(sparse_output=False)
        fit_array = np.array([[1],[2],[3]])
        self.enc.fit(fit_array)

    def __getitem__(self,x):
        if x+1 == self.len:
            h_start, h_stop = self.h5_file["event_hits_index"][x], len(self.h5_file["coords"])
        else :
            h_start, h_stop = self.h5_file["event_hits_index"][x], self.h5_file["event_hits_index"][x+1]
        n_hits = h_stop - h_start
        charge = charge_transform(self.h5_file["charge"][h_start:h_stop][:,None])
        coords = self.h5_file["coords"][h_start:h_stop] -self.h5_file["verPos"][x]
        coords = scale_coords(coords)
        vals = self.enc.transform(self.h5_file["labels"][h_start:h_stop])
        feats = np.concatenate([coords,charge], axis=1)
        # p2d = (0,0,0,64 - n_hits%64)
        # mask = torch.zeros(n_hits+ 64 - n_hits%64)
        # mask[:n_hits] = 1
        # t_coords, t_vals = torch.nn.functional.pad(torch.tensor(feats), p2d, value = 0), torch.nn.functional.pad(torch.tensor(vals), p2d, value = 0)
        # return {"coords": t_coords, "values": t_vals, "mask": mask }

        t_coords, t_vals = torch.tensor(feats), torch.tensor(vals)
       
        return {"coords": t_coords, "values": t_vals}
    
    def __len__(self):
        return self.len
    
class SFGD_tagging(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, use_h5 = True, num_workers = 15):
        super(SFGD_tagging, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_h5 = use_h5
        
    def prepare_data(self):
        ...
    
    def setup(self, stage):
        if self.use_h5:
            self.dataset = NodeCL_h5dataset(self.data_dir)
        else :
            self.dataset = NodeCL_dataset(self.data_dir)

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, lengths=[0.65,0.15,0.2])
    
    def train_dataloader(self):
        rand_sampler = RandomSampler(self.train_dataset)
        lenmatch_sampler = LenMatchBatchSampler(self.train_dataset, rand_sampler, batch_size= self.batch_size, drop_last=False)
        return DataLoader(self.train_dataset, batch_sampler=lenmatch_sampler, num_workers=self.num_workers, collate_fn=my_collate)
    
    def val_dataloader(self):
        rand_sampler = RandomSampler(self.val_dataset)
        lenmatch_sampler = LenMatchBatchSampler(self.val_dataset, rand_sampler, batch_size= self.batch_size, drop_last=False)
        return DataLoader(self.val_dataset, batch_sampler=lenmatch_sampler, num_workers=self.num_workers, collate_fn=my_collate)
    
    def test_dataloader(self):
        rand_sampler = RandomSampler(self.test_dataset)
        lenmatch_sampler = LenMatchBatchSampler(self.test_dataset, rand_sampler, batch_size= self.batch_size, drop_last=False)
        return DataLoader(self.test_dataset, batch_sampler=lenmatch_sampler, num_workers=self.num_workers, collate_fn=my_collate)
    
    def teardown(self, stage):
        ...

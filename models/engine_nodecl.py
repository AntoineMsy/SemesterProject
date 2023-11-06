from lightning.pytorch.utilities.types import OptimizerLRScheduler
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import StepLR,CyclicLR
from torchvision.transforms.transforms import RandomRotation
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
# import seaborn as sns
import pandas as pd
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
from lightning.pytorch.loggers.neptune import NeptuneLogger
from sklearn.metrics import f1_score, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.ticker as ticker

from models.transformer_encoder import TransformerSeg
from data.data_utils import inv_charge_transform, inv_coords_scale

class NodeClassificationEngine(pl.LightningModule):
    def __init__(self, model_name, model_kwargs, lr, epochs):
        super(NodeClassificationEngine,self).__init__()
        self.save_hyperparameters()
        self.example_input_array = (torch.Tensor(32, 5, 4), torch.Tensor(32, 5))
        self.model_kwargs = model_kwargs
        self.model_name = model_name
        self.lr = lr
        self.epochs = epochs

        valid_models = {"transformer_encoder" : TransformerSeg}
        
        self.model = valid_models[self.model_name](**self.model_kwargs)
        self.loss_fn = nn.CrossEntropyLoss()
        
    def forward(self,coords, mask):
        pred = self.model(coords, mask)
        return pred
    
    def step(self, batch):
        coords, target, mask = batch["coords"], batch["values"], batch["mask"]
        pred = self(coords, mask)
        loss = self.loss_fn(pred, target)
        return loss, pred, target
    
    def training_step(self, batch, batch_idx):
        loss, pred, target = self.step(batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, pred, target = self.step(batch)
        for i in range(len(batch)):
            true_labels = self.dataset.enc.inverse_transform(batch["values"][i,batch["mask"][i].bool()])
            pred_labels = self.dataset.enc.inverse_transform(batch["values"][i,batch["mask"][i].bool()])

            self.test_nhits.append(len(batch["values"][i,batch["mask"][i].bool()]))
            conf_mat = confusion_matrix(true_labels, pred_labels, normalize="true")

        charge = inv_charge_transform(batch["coords"][:,:,3])
        print(charge.size())
        torch.sum(charge, axis = 1)

        self.log("test_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss, pred, target
    
    def validation_step(self, batch, batch_idx):
        #enabling sync dist can slow down training, metrics from torchmetrics automatically handle this
        # When logging only on rank 0, don't forget to add
        # `rank_zero_only=True` to avoid deadlocks on synchronization.
        loss, pred, target = self.step(batch)
        self.log("validation_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss, pred, target
        
    def on_test_epoch_end(self):
        #Compute average loss, confusion matrix between classes and per event 
        print('test_epoch ended')

    def on_test_epoch_start(self):
        self.test_charge = torch.empty(0)
        self.test_nhits = torch.empty(0)
        self.test_conf_mat = torch.empty(0)

    def on_validation_epoch_end(self):
        print('val epoch ended')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        return [optimizer]

# # Find two GPUs on the system that are not already occupied
# trainer = Trainer(accelerator="cuda", devices=find_usable_cuda_devices(2))
# trainer = Trainer(log_every_n_steps=k)
# Trainer(precision="16-mixed")
# # Accumulate gradients for 7 batches
# trainer = Trainer(accumulate_grad_batches=7)
# # Enable Stochastic Weight Averaging using the callback
# trainer = Trainer(callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)])

# from lightning.pytorch.tuner import Tuner
# # Create a Tuner
# tuner = Tuner(trainer)

# # finds learning rate automatically
# # sets hparams.lr or hparams.learning_rate to that learning rate
# tuner.lr_find(model)

# class MNISTDataModule(pl.LightningDataModule):
#     def __init__(self, data_dir: str):
#         self.mnist = MNIST(data_dir, download=True, transform=T.ToTensor())

#     def train_loader(self):
#         return DataLoader(self.mnist, batch_size=128)


# model = Model(...)
# datamodule = MNISTDataModule("data/MNIST")

# trainer = Trainer(accelerator="gpu", devices=2, strategy="ddp_spawn")
# trainer.fit(model, datamodule)
# Trainer(fast_dev_run=True)
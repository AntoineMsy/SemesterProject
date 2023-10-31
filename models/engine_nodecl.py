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
from sklearn.metrics import f1_score,auc
import matplotlib.ticker as ticker

from sfgd_nodecl.models.transformer_encoder import Model

class NodeClassificationEngine(pl.LightningModule):
    def __init__(self, model_name, model_config):
        super(NodeClassificationEngine,self).__init__()
        self.save_hyperparameters()
    
    def forward(self):

    def step(self):
        return loss, logs
    def training_step(self):
        loss, logs = self.step()
        self.log_dir()
        return loss
    
    def test_step(self):
    
    def validation_step(self):
        #enabling sync dist can slow down training, metrics from torchmetrics automatically handle this
        # When logging only on rank 0, don't forget to add
        # `rank_zero_only=True` to avoid deadlocks on synchronization.
        self.log("validation_loss", loss, on_step=True, on_epoch=True, sync_dist=True)
    def training_epoch_end(self):
    
    def test_epoch_end(self):
    
    def validation_epoch_end(self):


# Find two GPUs on the system that are not already occupied
trainer = Trainer(accelerator="cuda", devices=find_usable_cuda_devices(2))
trainer = Trainer(log_every_n_steps=k)
Trainer(precision="16-mixed")
# Accumulate gradients for 7 batches
trainer = Trainer(accumulate_grad_batches=7)
# Enable Stochastic Weight Averaging using the callback
trainer = Trainer(callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)])

from lightning.pytorch.tuner import Tuner
# Create a Tuner
tuner = Tuner(trainer)

# finds learning rate automatically
# sets hparams.lr or hparams.learning_rate to that learning rate
tuner.lr_find(model)

class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str):
        self.mnist = MNIST(data_dir, download=True, transform=T.ToTensor())

    def train_loader(self):
        return DataLoader(self.mnist, batch_size=128)


model = Model(...)
datamodule = MNISTDataModule("data/MNIST")

trainer = Trainer(accelerator="gpu", devices=2, strategy="ddp_spawn")
trainer.fit(model, datamodule)
Trainer(fast_dev_run=True)
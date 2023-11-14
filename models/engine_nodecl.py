from lightning.pytorch.utilities.types import OptimizerLRScheduler
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ExponentialLR

# import matplotlib
# matplotlib.use('Agg')

from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.ticker as ticker

from models.transformer_encoder import TransformerSeg
from data.data_utils import inv_charge_transform, inv_scale_coords
from data.plotting_utils import *

class NodeClassificationEngine(pl.LightningModule):
    def __init__(self, model_name, model_kwargs, lr, use_weighted_loss = False):
        super(NodeClassificationEngine,self).__init__()
        self.save_hyperparameters()
        self.example_input_array = (torch.Tensor(32, 5, 4), torch.Tensor(32, 5))
        self.model_kwargs = model_kwargs
        self.model_name = model_name
        self.lr = lr

        valid_models = {"transformer_encoder" : TransformerSeg}
        
        self.model = valid_models[self.model_name](**self.model_kwargs)
        if use_weighted_loss:
            self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1 + np.log(37),1,1 + np.log(3)]))
        else:
            self.loss_fn = nn.CrossEntropyLoss()
        
        self.softmax = torch.nn.Softmax(dim=-1)
        
    def forward(self,coords, mask):
        pred = self.model(coords, mask)
        return pred
    
    def step(self, batch):
        coords, target, mask = batch["coords"], batch["values"], batch["mask"]
        pred = self(coords, mask)
        loss = self.loss_fn(pred.transpose(1,2), target.transpose(1,2))
        return loss, pred, target
    
    def training_step(self, batch, batch_idx):
        loss, pred, target = self.step(batch)
        self.log("train_loss", loss, on_step=True, rank_zero_only=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, pred, target = self.step(batch)
        class_pred  = torch.argmax(pred, 2) + 1
        for i in range(len(batch["coords"])):
            true_labels = torch.from_numpy(self.trainer.datamodule.dataset.enc.inverse_transform(batch["values"][i,batch["mask"][i].bool()].cpu().detach()))
            pred_labels = class_pred[i,batch["mask"][i].bool()].cpu().detach()
            conf_mat = torch.tensor(confusion_matrix(true_labels, pred_labels, labels = [1,2,3], normalize="true"))[None,:,:]

            self.test_conf_mat = torch.cat((self.test_conf_mat, conf_mat))
            self.test_preds = torch.cat((self.test_preds, pred_labels))
            self.test_vals = torch.cat((self.test_vals, true_labels))
            self.test_nhits = torch.cat((self.test_nhits, torch.tensor([len(batch["values"][i,batch["mask"][i].bool()])])))
            self.test_softmax = torch.cat((self.test_softmax,pred[i,batch["mask"][i].bool()].cpu().detach()))
        

        charge = inv_charge_transform(batch["coords"][:,:,3].cpu().detach())
        self.test_charge = torch.cat((self.test_charge,torch.sum(charge, axis = 1)))
        self.log("test_loss", loss, on_epoch=True, rank_zero_only=True)
        return loss, pred, target
    
    def validation_step(self, batch, batch_idx):
        #enabling sync dist can slow down training, metrics from torchmetrics automatically handle this
        # When logging only on rank 0, don't forget to add
        # `rank_zero_only=True` to avoid deadlocks on synchronization.
        loss, pred, target = self.step(batch)
        self.val_loss_list = torch.cat((self.val_loss_list, loss.cpu().detach()[None]))
        self.log("validation_loss", loss, on_epoch = True, rank_zero_only=True )
        return loss, pred, target
        
    def on_test_epoch_end(self):
        #Compute average loss, confusion matrix between classes and per event
        print(self.test_charge.size(), self.test_softmax.size(), self.test_nhits.size())
        conf_mat = torch.tensor(confusion_matrix(self.test_vals, self.test_preds, labels = [1,2,3], normalize="true"))[None,:,:]
        conf_mat_T = torch.tensor(confusion_matrix(self.test_preds, self.test_vals, labels = [1,2,3], normalize="true"))[None,:,:]
       
        print(conf_mat)
        print(conf_mat_T)
       
        s_dir = self.logger.log_dir
        plot_conf_mat(conf_mat, save_plot=True, save_dir= s_dir+"/conf_mat.png")
        plot_binned_efficiency(self.test_conf_mat, self.test_charge, val_name="charge", bin_num=50, save_plot=True, save_dir = s_dir+"/charge_plot.png" )
        plot_binned_efficiency(self.test_conf_mat, self.test_nhits, val_name="n_hits", bin_num=50, save_plot=True, save_dir = s_dir+"/nhits_plot.png" )
        plot_discriminators(self.test_softmax, self.test_vals, save_dir = s_dir+"/softmax.png")
        print('test_epoch ended')

    def on_test_epoch_start(self):
        self.test_charge = torch.empty(0)
        self.test_nhits = torch.empty(0)
        self.test_conf_mat = torch.empty(0)
        self.test_preds = torch.empty(0)
        self.test_vals = torch.empty(0)
        self.test_softmax = torch.empty(0)
    
    def on_validation_epoch_start(self):
        self.val_loss_list = torch.empty(0)

    def on_validation_epoch_end(self):
        print('val epoch ended')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        lr_warmup = LinearLR(optimizer, start_factor=0.001, end_factor=1, total_iters=600)
        def lr_foo(epoch):
            if epoch < self.hparams.warm_up_step:
                # warm up lr
                lr_scale = 0.1 ** (self.hparams.warm_up_step - epoch)
            else:
                lr_scale = 0.95 ** epoch

            return lr_scale

        scheduler = SequentialLR(optimizer,
            [LinearLR(optimizer,0.001,1,total_iters=1000), ExponentialLR(optimizer,0.9999)], milestones=[1000]
        )

        return [optimizer], {"scheduler" : scheduler, "interval": "step"}


# trainer = Trainer(log_every_n_steps=k)
# Trainer(precision="16-mixed")

# # Enable Stochastic Weight Averaging using the callback
# trainer = Trainer(callbacks=[StochasticWeightAveraging(swa_lrs=1e-2)])

# from lightning.pytorch.tuner import Tuner
# # Create a Tuner
# tuner = Tuner(trainer)

# # finds learning rate automatically
# # sets hparams.lr or hparams.learning_rate to that learning rate
# tuner.lr_find(model)


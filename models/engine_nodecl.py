from lightning.pytorch.utilities.types import OptimizerLRScheduler
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import SequentialLR, LinearLR, ExponentialLR, CosineAnnealingWarmRestarts

from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import f1_score, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.ticker as ticker

from models.transformer_encoder import *
from data.data_utils import inv_charge_transform, inv_scale_coords
from data.plotting_utils import *

class NodeClassificationEngine(pl.LightningModule):
    # Main Model class, encapsulates train, test and validation
    # model_kwargs corresponds to the arguments of the chosen model in the valid_models dictionary
    #  You can choose how to weight the CE Loss too
    def __init__(self, model_name, model_kwargs, lr=1.0e-3, weight = [1,1,1], use_weighted_loss = False):
        super(NodeClassificationEngine,self).__init__()
        self.save_hyperparameters()
        self.example_input_array = (torch.Tensor(32, 5, 4), torch.Tensor(32, 5))
        self.model_kwargs = model_kwargs
        self.model_name = model_name
        self.lr = lr

        valid_models = {"transformer_encoder" : TransformerSeg_pre, "baseline" : TransformerSeg, "v0": TransformerSeg_v0, "v1": TransformerSeg_v1}
        
        self.model = valid_models[self.model_name](**self.model_kwargs)

        if use_weighted_loss:
            self.loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weight)) 
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        self.softmax = torch.nn.Softmax(dim=-1)
        
    def forward(self,coords, mask):
        pred = self.model(coords, mask)
        return pred
    
    def step(self, batch):
        coords, target, mask = batch["coords"], batch["values"], batch["mask"]
        pred = self(coords, mask)
        # Use mask to convert the predictions into a 2D output and then apply the loss function
        loss = self.loss_fn(pred[mask.bool()],target[mask.bool()])
        return loss, pred, target
    
    def training_step(self, batch, batch_idx):
        loss, pred, target = self.step(batch)
        self.log("train_loss", loss, on_step=True, rank_zero_only=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        # Logs the prediction for each event separately to make statistics
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
            self.test_softmax = torch.cat((self.test_softmax,torch.softmax(pred[i,batch["mask"][i].bool()],dim=-1).cpu().detach()))
            charge = inv_charge_transform(batch["coords"][i,:,3][batch["mask"][i].bool()].cpu().detach())
            
            self.test_charge = torch.cat((self.test_charge,torch.sum(charge,dim=0,keepdim=True)))

        self.log("test_loss", loss, on_epoch=True, rank_zero_only=True)
        return loss, pred, target
    
    def validation_step(self, batch, batch_idx):
        loss, pred, target = self.step(batch)
        class_pred  = torch.argmax(pred, 2) + 1
        self.val_loss_list = torch.cat((self.val_loss_list, loss.cpu().detach()[None]))

        true_labels = torch.from_numpy(self.trainer.datamodule.dataset.enc.inverse_transform(batch["values"][batch["mask"].bool(),:].cpu().detach()))
        pred_labels = class_pred[batch["mask"].bool()].cpu().detach()

        self.val_preds = torch.cat((self.val_preds, pred_labels))
        self.val_vals = torch.cat((self.val_vals, true_labels))

        self.log("validation_loss", loss, on_epoch = True, rank_zero_only=True )

        return loss, pred, target
        
    def on_test_epoch_end(self):
        #Compute average loss, confusion matrix between classes and per event from the test_step logging
        print(self.test_charge.size(), self.test_softmax.size(), self.test_nhits.size(), self.test_vals.size())
        conf_mat = torch.tensor(confusion_matrix(self.test_vals, self.test_preds, labels = [1,2,3], normalize="true"))[None,:,:]
        conf_mat_T = torch.tensor(confusion_matrix(self.test_preds, self.test_vals, labels = [1,2,3], normalize="true"))[None,:,:]
        
        print(conf_mat)
        print(conf_mat_T)

        s_dir = self.logger.log_dir

        plot_conf_mat(conf_mat, save_plot=True, save_dir= s_dir+"/conf_mat_eff.png")
        plot_conf_mat(conf_mat_T, save_plot=True, save_dir=s_dir + "/conf_mat_pur.png", mode="pur")

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
        self.val_preds = torch.empty(0)
        self.val_vals = torch.empty(0)

    def on_validation_epoch_end(self):
        # Metrics for validation :
        #   - Purity and efficiency for each class
        #   - Macro and Micro F1 score
        conf_mat = torch.tensor(confusion_matrix(self.val_vals, self.val_preds, labels = [1,2,3], normalize="true"))
        conf_mat_T = torch.tensor(confusion_matrix(self.val_preds, self.val_vals, labels = [1,2,3], normalize="true"))

        self.log("Macro-F1", f1_score(self.val_vals, self.val_preds, average="macro"))
        self.log("Micro-F1", f1_score(self.val_vals, self.val_preds, average="micro"))

        # print(compute_class_weight("balanced", [1,2,3], self.val_vals))
        # self.log("Weighted-F1", f1_score(self.val_vals, self.val_preds, sample_weight=compute_class_weight("balanced", classes=[1,2,3], y=self.val_vals), average="weighted"))
        self.log_dict({"MP-eff": conf_mat[0,0], "SP-eff" : conf_mat[1,1], "N-eff" : conf_mat[2,2]}, on_epoch = True, rank_zero_only=True )
        self.log_dict({"MP-pur": conf_mat_T[0,0], "SP-pur" : conf_mat_T[1][1], "N-pur" : conf_mat_T[2][2]}, on_epoch = True, rank_zero_only=True )
      
        print('val epoch ended')

    def configure_optimizers(self):
        # Using an exponential decay scheduler with a warmup of 1000 steps
        optimizer = torch.optim.Adam(self.parameters(), lr = self.lr)
        # optimizer = Lamb(self.parameters(),lr = self.lr)
        # optimizer = torch.optim.AdamW(self.parameters(), lr = self.lr)
        # scheduler = SequentialLR(optimizer,
        #     [LinearLR(optimizer,0.001,1,total_iters=1000), CosineAnnealingWarmRestarts(optimizer,T_0 = 1500)], milestones=[1000]
        # )
        scheduler = SequentialLR(optimizer,
            [LinearLR(optimizer,0.001,1,total_iters=1000), ExponentialLR(optimizer,0.9999)], milestones=[1000]
        )   
        return [optimizer], {"scheduler" : scheduler, "interval": "step"}

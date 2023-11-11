import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import ConfusionMatrixDisplay

# import seaborn as sns

#plot:
# - binned  average efficiency over #hits and charge (one func)
def plot_conf_mat(conf_mat_list, save_plot = True, save_dir = "conf_mat.png"):
    conf_display = ConfusionMatrixDisplay(np.array(torch.mean(conf_mat_list, dim = 0)), display_labels=["multi_P", "single_P", "noise"])
    conf_display.plot()
    if save_plot:
        plt.savefig(save_dir)
    plt.clf()

def plot_binned_efficiency(conf_mat_list, bin_val, val_name = r"$Charge [a.u.]$", bin_num = 5, save_plot = True, save_dir = "binned_efficiency_plot.png"):
    #takes a lists of 3x3 confusion matrix and plot its values (here efficiency) 
    # in a binned way over a binning value (n_hits, charge...)

    binning_weights = np.array([[conf_mat_list[k][i][i] for k in range(len(bin_val))] for i in range(3)])
    binning = np.linspace(0, max(bin_val), bin_num)
    indices = np.digitize(bin_val, binning)
    for k in range(3):
        avg_bin = [np.average([binning_weights[k][np.where(indices == i)[0]]])  for i in range(1,bin_num+1)]
        plt.step(binning, avg_bin, where='post', linestyle="-")

    plt.xlabel(val_name)
    plt.ylabel("efficiency")
    plt.title("Mean binned efficiency")
    if save_plot:
        plt.savefig(save_dir)
    plt.clf()
    # plt.hist(bin_val, weights = binning_weights)

# def display_3D(evt):
#     fig = plt.figure(figsize=(10,10))
#     ax = fig.add_subplot(projection='3d')
#     c_list = ["yellow", "red", "blue"]
#     labels = ["multi", "single", "other"]
#     for i in range(1,4):
#         cat_coords = evt["c"][(evt["y"]==i).squeeze(1),:]
#         ax.scatter(cat_coords[:,0], cat_coords[:,1], cat_coords[:,2], color = c_list[i-1], edgecolor = "black", alpha = 1, label = labels[i-1])
#     ax.legend()
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")

# def display_3D_h5(evt_id):
#     fig = plt.figure(figsize=(10,10))
#     ax = fig.add_subplot(projection='3d')
#     c_list = ["yellow", "red", "blue"]
#     labels = ["multi", "single", "other"]
#     h_start, h_stop = hf["event_hits_index"][evt_id], hf["event_hits_index"][evt_id+1]

#     coords = hf["coords"][h_start:h_stop]
#     vals = hf["labels"][h_start:h_stop]

#     for i in range(1,4):
#         cat_coords =coords[(vals==i).squeeze(1),:]
#         ax.scatter(cat_coords[:,0], cat_coords[:,1], cat_coords[:,2], color = c_list[i-1], edgecolor = "black", alpha = 1, label = labels[i-1])
#     ax.legend()
#     ax.set_xlabel("X")
#     ax.set_ylabel("Y")
#     ax.set_zlabel("Z")

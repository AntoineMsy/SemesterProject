import numpy as np
import h5py
import os
import sys
import tqdm
import argparse

"""
Converts a folder of npz files into a single h5 file for faster data reading and easier manipulation
Use in the command line via the argparser below
You can filter the dataset based on number of hits
"""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_folder", help="Path to the input folder", default="/scratch2/sfgd/sparse_data_genie_fhc_numu_hittag/")
   
    parser.add_argument("-o", "--output_file", help="Path to the output file", default="out.h5")
    parser.add_argument("-l", "--max_evt_length", type=int, default=5000)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    config = get_args()
    data_path = config.input_folder
    N_files = len(os.listdir(data_path))
    f = h5py.File(config.output_file, 'w')

    total_rows = 0
    total_hits = 0
    
    max_event_length = config.max_evt_length

    print("counting events and hits, in files")
    good_idx = []
    for i in tqdm.tqdm(range(N_files)):
        npz_file = np.load(data_path + "event%d.npz"%i)

        coords = npz_file["c"]
        if len(coords) <= max_event_length:
            good_idx.append(i)
            total_rows += 1
            total_hits += len(coords)

    """
    Initiate the h5py dataset
    """

    dset_hit_coords = f.create_dataset("coords",
                                    shape=(total_hits, 3),
                                    dtype=np.float32)

    dset_hit_charge = f.create_dataset("charge",
                                    shape=(total_hits),
                                    dtype=np.float32)

    dset_hit_labels = f.create_dataset("labels",
                                    shape=(total_hits, 1),
                                    dtype=np.float32)

    dset_vertex_pos = f.create_dataset("verPos",
                                    shape=(total_rows, 3),
                                    dtype=np.float32)

    dset_event_hit_index = f.create_dataset("event_hits_index",
                                                shape=(total_rows,),
                                                dtype=np.int64) 
    offset = 0
    offset_next = 0
    hit_offset = 0
    hit_offset_next = 0

    print("Creating dataset")
    
    for i in tqdm.tqdm(good_idx):
        npz_file = np.load(data_path + "event%d.npz"%i)
        dset_event_hit_index[offset] = hit_offset
        coords = npz_file["c"]
        cat = npz_file["y"]
        time = npz_file["x"][:,0]
        charge = npz_file["x"][:,1]
        vpos = npz_file["verPos"]

        hit_offset_next += len(coords)

        dset_hit_coords[hit_offset:hit_offset_next] = coords
        dset_hit_charge[hit_offset:hit_offset_next] = charge
        dset_hit_labels[hit_offset:hit_offset_next] = cat
        dset_vertex_pos[offset] = vpos
        offset+=1
        hit_offset = hit_offset_next

    f.close()
    print("file saved")
import numpy as np
import h5py
import os
import sys
import tqdm

data_path = "/scratch2/sfgd/sparse_data_genie_fhc_numu_hittag/"
N_files = len(os.listdir(data_path))
f = h5py.File("out.h5", 'w')

total_rows = 0
total_hits = 0
min_hits = 1

print("counting events and hits, in files")

for i in tqdm.tqdm(range(N_files)):
    npz_file = np.load(data_path + "event%d.npz"%i)

    coords = npz_file["c"]

    total_rows += 1
    total_hits += len(coords)

    if i == 0 :
        min_hits = len(coords)
    elif len(coords) < min_hits:
        min_hits = len(coords)
    
        
    
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

for i in tqdm.tqdm(range(N_files)):
    npz_file = np.load(data_path + "event%d.npz"%i)
    dset_event_hit_index[i] = hit_offset
    coords = npz_file["c"]
    cat = npz_file["y"]
    time = npz_file["x"][:,0]
    charge = npz_file["x"][:,1]
    vpos = npz_file["verPos"]

    hit_offset_next += len(coords)

    dset_hit_coords[hit_offset:hit_offset_next] = coords
    dset_hit_charge[hit_offset:hit_offset_next] = charge
    dset_hit_labels[hit_offset:hit_offset_next] = cat
    dset_vertex_pos[i] = vpos

    hit_offset = hit_offset_next

f.close()
print("file saved")
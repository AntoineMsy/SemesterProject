### Hit Identification inside SuperFGD with Transformers

## Basic use
- To run the baseline configuration, run:
`python main.py -f "config/baseline.yaml"`

 -To run in test mode, simply add a `-t` argument above.

 ## Overview
- config holds the different configuration files for training and testing.
- data holds several files for data handling, Lightning DataModule creation and preprocessing.
- models contains the main engine in `engine_nodecl.py` and the different variants of transformer encoders that have been tested in `transformer_encoder.py`
- notebooks gives notebooks to illustrate the work. In `sfgd_eda.ipynb` you'll basic data analysis on the dataset. `model_inference.ipynb` displays attention maps and the model predictions compared to the ground truth. `umap_embedding.ipynb` applies the UMAP dimensionality reduction on the last layer output of the model (before the linear classifier).

 

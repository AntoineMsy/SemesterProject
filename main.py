import lightning.pytorch as pl
from data.sfgd_datamodules import SFGD_tagging
from models.engine_nodecl import NodeClassificationEngine
from lightning.pytorch.accelerators import find_usable_cuda_devices

data_path = data_path = "/scratch2/sfgd/sparse_data_genie_fhc_numu_hittag/"
datamod = SFGD_tagging(data_dir=data_path, batch_size=32)

model_config = {"d_model" : 64, "d_ff": 32, "num_heads" : 4, "num_layers" : 5, "in_features" : 4, "out_features" : 3}
model = NodeClassificationEngine("transformer_encoder", model_kwargs= model_config, lr = 1e-3, epochs=2)

trainer = pl.Trainer(accelerator="cuda", devices=find_usable_cuda_devices(2), precision="16-mixed")
trainer.fit(model=model, datamodule=datamod)



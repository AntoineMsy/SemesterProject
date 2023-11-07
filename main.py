import lightning.pytorch as pl
from data.sfgd_datamodules import SFGD_tagging
from models.engine_nodecl import NodeClassificationEngine
from lightning.pytorch.accelerators import find_usable_cuda_devices
import argparse
import yaml
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import DeviceStatsMonitor
from lightning.pytorch.callbacks import ModelCheckpoint

def get_args():
    parser = argparse.ArgumentParser(description="Parse YAML file with argparse")
    parser.add_argument("-f", "--file", required=True, help="Path to the YAML file")
    parser.add_argument("-n", "--name", required=False, help="Run Name for Tensorboard")
    parser.add_argument("-t", "--test_mode", action="store_true", help="Enable test mode")

    args = parser.parse_args()
    return args

def parse_yaml(file_path):
    with open(file_path, "r") as file:
        try:
            data = yaml.safe_load(file)
            return data
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            return None

if __name__ == "__main__":
    config = get_args()
    yaml_data = parse_yaml(config.file)
    data_path = yaml_data["data_dir"]

    datamod = SFGD_tagging(data_dir= data_path, batch_size= yaml_data["batch_size"])

    model = NodeClassificationEngine("transformer_encoder", model_kwargs= yaml_data["model_config"], lr = yaml_data["learning_rate"])
    if config.name:
        run_name = config.name
        logger = TensorBoardLogger("tb_logs", name="my_model")
    else : 
        logger = TensorBoardLogger("tb_logs", name="my_model")

    trainer_args = {"accelerator" : "cuda", 
                    "devices": find_usable_cuda_devices(yaml_data["num_devices"]), 
                    "precision" : "16-mixed", 
                    **yaml_data["trainer_config"]
                    }
    callbacks = [EarlyStopping(monitor = "mean_val_loss", mode = "min")]
    if config.test_mode :
        print("RUNNING IN DEV MODE")
        trainer = pl.Trainer(fast_dev_run= 10, **trainer_args)
    
    else :
        trainer = pl.Trainer(**trainer_args)
    
    trainer.fit(model=model, datamodule=datamod)
    trainer.test(model=model, datamodule=datamod)



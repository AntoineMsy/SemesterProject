import lightning.pytorch as pl
from data.sfgd_datamodules import SFGD_tagging
from data.data_utils import parse_yaml
from models.engine_nodecl import NodeClassificationEngine
from lightning.pytorch.accelerators import find_usable_cuda_devices
import argparse
from lightning.pytorch.loggers.tensorboard import TensorBoardLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import DeviceStatsMonitor, LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint
def get_args():
    parser = argparse.ArgumentParser(description="Parse YAML file with argparse")
    parser.add_argument("-f", "--file", required=True, help="Path to the YAML file")
    parser.add_argument("-n", "--name", required=False, help="Run Name for Tensorboard")
    parser.add_argument("-t", "--test_mode", action="store_true", help="Enable test mode")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    config = get_args()
    yaml_data = parse_yaml(config.file)
    data_path = yaml_data["data_dir"]

    datamod = SFGD_tagging(data_dir= data_path, batch_size= yaml_data["batch_size"])

    model = NodeClassificationEngine(**yaml_data['engine'])
    if config.name:
        run_name = config.name
        logger = TensorBoardLogger("tb_logs", name=run_name)
    else : 
        logger = TensorBoardLogger("tb_logs", name="my_model")

    
    if "num_devices" in list(yaml_data.keys()):
        dev = find_usable_cuda_devices(yaml_data["num_devices"])
    else :
        dev = yaml_data['devices_names']

    callbacks = [EarlyStopping(monitor = "validation_loss", patience = 10, log_rank_zero_only=True),
                 LearningRateMonitor(logging_interval="step")]

    trainer_args = {"accelerator" : "cuda", 
                    "devices": dev, 
                    "precision" : "16-mixed", 
                    "logger" : logger,
                    "callbacks" : callbacks,
                    **yaml_data["trainer_config"]
                    }
    

    if config.test_mode :
        print("RUNNING IN DEV MODE")
        trainer = pl.Trainer(fast_dev_run= 2, **trainer_args)
    
    else :
        trainer = pl.Trainer(**trainer_args)

    if "fit" in yaml_data["tasks"]:
        trainer.fit(model=model, datamodule=datamod)
        
    if "test" in yaml_data["tasks"]:
        if "ckpt_path" in yaml_data.keys():
            model = NodeClassificationEngine.load_from_checkpoint(yaml_data["ckpt_path"])

        trainer.test(model=model, datamodule=datamod)



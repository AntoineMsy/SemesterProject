import lightning.pytorch as pl
from data.sfgd_datamodules import SFGD_tagging
from models.engine_nodecl import NodeClassificationEngine
from lightning.pytorch.accelerators import find_usable_cuda_devices
import argparse
import yaml

def get_args():
    parser = argparse.ArgumentParser(description="Parse YAML file with argparse")
    parser.add_argument("-f", "--file", required=True, help="Path to the YAML file")
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

    model = NodeClassificationEngine("transformer_encoder", model_kwargs= yaml_data["model_config"], lr = yaml_data["learning_rate"], epochs=2)

    trainer = pl.Trainer(accelerator="cuda", devices=find_usable_cuda_devices(yaml_data["num_devices"]), precision="bf16-mixed", max_epochs=yaml_data["max_epochs"])
    trainer.fit(model=model, datamodule=datamod)



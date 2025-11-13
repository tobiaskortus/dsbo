# flake8: noqa: E402
import os
import pathlib
import sys
import numpy as np
from typing import Union

import pandas as pd
import torch
from torch.utils.data import Dataset

# Add the directory containing this script to sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from dataset import ReconstructionDataset
from model import Reconstruction
from validation import ReconstructionValidation


def pre_train(model: Reconstruction, dataset: Dataset, n_epochs: int):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)

    print("Reconstruction: Pre-training 0")
    model.train_model(dataset, batch_size=512, n_epochs=n_epochs, lr=0.001)

    print("Reconstruction: Pre-training 1")
    model.train_model(dataset, batch_size=1024, n_epochs=n_epochs, lr=0.001)
    model.to('cpu')


def train(
    input_df_path: Union[str, os.PathLike],
    output_df_path: Union[str, os.PathLike],
    isVal: bool,
    model_path: Union[str, os.PathLike],
):
    print(input_df_path, output_df_path, model_path)
    simulation_df: pd.DataFrame = pd.read_parquet(input_df_path)
    isval = False

    if isVal:
        #reco_model: Reconstruction = torch.load(os.path.join(results_dir, "reco_model"))
        #reco_dataset = ReconstructionDataset(simulation_df, means=reco_model.means, stds=reco_model.stds)

        #validator = ReconstructionValidation(reco_model)
        #output_df_val = validator.validate(reco_dataset)
        #output_df_val.to_parquet(output_df_path)
        pass
    else:
        n_epochs_pre = 24
        n_epochs_main = 40
        reco_dataset = ReconstructionDataset(simulation_df)
        reco_model = Reconstruction(*reco_dataset.shape, reco_dataset.means, reco_dataset.stds)
        pre_train(reco_model, reco_dataset, n_epochs_pre)

        # Reconstruction training:
        reco_model.to("cuda" if torch.cuda.is_available() else "cpu")
        reco_model.train_model(reco_dataset, batch_size=256,  n_epochs=n_epochs_main // 4, lr=0.003)
        reco_model.train_model(reco_dataset, batch_size=1024, n_epochs=n_epochs_main // 2, lr=0.001)
        reco_model.train_model(reco_dataset, batch_size=1024, n_epochs=n_epochs_main // 2, lr=0.0003)

        validator = ReconstructionValidation(reco_model)
        output_df_val = validator.validate(reco_dataset)
        output_df_val.to_parquet(output_df_path)
        #torch.save(reco_model, model_path)


if __name__ == "__main__":
    input_df_path = sys.argv[1]
    output_df_path = sys.argv[2]
    isVal = sys.argv[3].strip().lower() == "true"
    model_path = sys.argv[4]
    train(input_df_path, output_df_path, isVal, model_path)

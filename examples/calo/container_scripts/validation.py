import os
import sys
from typing import Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Add the directory containing this script to sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from model import Reconstruction, ReconstructionDataset

class ReconstructionValidation():
    """ Validate a given instance of the Reconstruction model
    """
    def __init__(
            self,
            reco_model: Reconstruction,
            ):
        """
        Initializes the Validation object for a given Reconstruction model.
        Args:
            reco_model (Reconstruction): The reconstruction model to be used.
        Attributes:
            device (torch.device): The device to be used for computation, either 'cuda' if a GPU is available or 'cpu'.
            """
        self.reco_model = reco_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def validate(
            self,
            val_dataset: ReconstructionDataset,
            batch_size: int = 512,
            ) -> pd.DataFrame:
        """ Apply the Reconstruction model on the validation dataset `val_dataset` and concatenate the
        results with that dataset, adding the columns ("Loss", "Reco_loss") and ("Reconstructed", "true_energy")
        as a multi-column index.
        Args:
            val_dataset (ReconstructionDataset): A valid dataset trained on simulation data distinct from the
            original Reconstruction model's training dataset.
            batch_size (int): Batch size for validation
        """
        val_result, val_loss, _ = self.reco_model.apply_model_in_batches(val_dataset, batch_size=batch_size)

        validation_df = pd.DataFrame({"true_energy": val_result})
        validation_df = pd.concat({"Reconstructed": validation_df}, axis=1)
        loss_df_val = pd.DataFrame({"Reco_loss": val_loss.tolist()})
        loss_df_val = pd.concat({"Loss": loss_df_val}, axis=1)
        output_df_val: pd.DataFrame = pd.concat([val_dataset.df, validation_df, loss_df_val], axis=1)
        return output_df_val
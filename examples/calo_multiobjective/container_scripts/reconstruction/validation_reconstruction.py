from typing import Union

from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .model import Reconstruction, ReconstructionDataset
from minipandas import MiniFrame

matplotlib.use("agg")


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
        val_result, val_losses, val_loss = self.reco_model.apply_model_in_batches(val_dataset, batch_size=batch_size)
        #output_mf = deepcopy(val_dataset.mf)
        #del output_mf._data['Inputs']

        #reco_mf = MiniFrame({"true_energy": val_result})
        #output_mf.add('Reconstructed',reco_mf)
        #output_mf.add('Loss',MiniFrame({'Reco_loss':val_loss}))
        return val_loss

    @classmethod
    def plot(cls, validation_df: pd.DataFrame, fig_savepath: Union[str, None]) -> None:

        reco = validation_df["Reconstructed"]["true_energy"]
        true = validation_df["Targets"]["true_energy"]

        fig, axs = plt.subplots(ncols=2,figsize=(9,4))
        bins = np.linspace(
            min(reco.min(),true.min()),
            max(reco.max(),true.max()),
            40 + 1,
        )

        axs[0].hist(true, bins=bins, label=r"$E_\text{true}$" + " (Simulation)", histtype="step", color="green")
        axs[0].hist(reco, bins=bins, label=r"$E_\text{reco}$" + " (Reconstruction)", histtype="step", color="blue")
        axs[0].set_xlabel("Energy [GeV]")
        axs[0].set_ylabel(f"Counts / ({(bins[1] - bins[0]):.2f} GeV)")
        axs[0].set_yscale('log')
        y_min,y_max = axs[0].get_ylim()
        axs[0].set_ylim(1e-1,y_max*20)
        axs[0].legend()

        h = axs[1].hist2d(
            true,
            reco,
            bins = bins,
            norm = matplotlib.colors.LogNorm(vmin=1),
        )
        axs[1].set_xlabel(r"$E_\text{true}$" + " (Simulation) GeV")
        axs[1].set_ylabel(r"$E_\text{reco}$" + " (Reconstruction) GeV")
        fig.colorbar(h[3], ax=axs[1])
        plt.tight_layout()

        if fig_savepath is not None:
            plt.savefig(fig_savepath,dpi=600)
            plt.close()

            print(f"Validation Plots Saved to '{fig_savepath}'")
            return None
        else:
            return fig, axs, bins

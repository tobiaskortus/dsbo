from typing import Union

from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np
import pandas as pd
import torch

from .model import Classification, ClassificationDataset
from minipandas import MiniFrame

matplotlib.use("agg")


class ClassificationValidation():
    """ Validate a given instance of the Classification model
    """
    def __init__(
            self,
            class_model: Classification,
            ):
        """
        Initializes the Validation object for a given Classification model.
        Args:
            class_model (Classification): The reconstruction model to be used.
        Attributes:
            device (torch.device): The device to be used for computation, either 'cuda' if a GPU is available or 'cpu'.
            """
        self.class_model = class_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def validate(
            self,
            val_dataset: ClassificationDataset,
            batch_size: int = 512,
            ) -> pd.DataFrame:
        """ Apply the Classification model on the validation dataset `val_dataset` and concatenate the
        results with that dataset, adding the columns ("Loss", "Reco_loss") and ("Reconstructed", "true_energy")
        as a multi-column index.
        Args:
            val_dataset (ClassificationDataset): A valid dataset trained on simulation data distinct from the
            original Classification model's training dataset.
            batch_size (int): Batch size for validation
        """
        val_dataset.augmentations = []
        val_result, val_losses, val_loss = self.class_model.apply_model_in_batches(val_dataset, batch_size=batch_size)
        #output_mf = deepcopy(val_dataset.mf)
        #del output_mf._data['Inputs']

        #reco_mf = MiniFrame(
        #    {
        #        f"true_logits_{i}": val_result[:,i]
        #        for i in range(val_result.shape[-1])
        #    }
        #)
        #output_mf.add('Reconstructed',reco_mf)
        #output_mf.add('Loss',MiniFrame({'Class_loss':val_loss}))
        return val_loss

    @classmethod
    def plot(cls, validation_mf: pd.DataFrame, fig_savepath: Union[str, None]) -> None:

        columns = list(validation_mf["Classes"].keys())
        names = [col.replace('contains:','') for col in columns]
        reco = np.concatenate(
            [
                validation_mf["Reconstructed"][f"true_logits_{i}"].reshape(-1,1)
                for i in range(len(columns))
            ],
            axis = 1,
        ) # logits
        true = np.concatenate(
            [
                validation_mf["Classes"][col].reshape(-1,1)
                for col in columns
            ],
            axis = 1,
        ) # true class 0/1

        fig, axs = plt.subplots(ncols=2,nrows=len(names),figsize=(9,len(names)*4))
        if axs.ndim == 1:
            axs = axs.reshape(1,-1)
        colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(names)))
        for i, name in enumerate(names):
            bins = np.linspace(reco[:,i].min(),reco[:,i].max(), 40+1)
            axs[i,0].hist(reco[:,i][true[:,i]==0], bins=bins, label=f"Class {name} (Reconstruction) [class=0]", histtype="step", color='blue', linestyle='dashed')
            axs[i,0].hist(reco[:,i][true[:,i]==1], bins=bins, label=f"Class {name} (Reconstruction) [class=1]", histtype="step", color='blue', linestyle='dotted')
            axs[i,0].set_xlabel("Class logits")
            axs[i,0].set_ylabel(f"Counts / ({(bins[1] - bins[0]):.2f})")
            axs[i,0].set_yscale('log')
            y_min,y_max = axs[i,0].get_ylim()
            axs[i,0].set_ylim(1e-1,y_max*20)
            axs[i,0].legend()

            fpr, tpr, _= roc_curve(true[:,i].astype(np.float32),reco[:,i])

            axs[i,1].plot(
                tpr,
                fpr,
                label = f'AUC = {auc(fpr,tpr):.5f}',
                linewidth = 2,
                color = 'royalblue',
            )
            axs[i,1].set_xlabel('TPR')
            axs[i,1].set_ylabel('FPR')
            axs[i,1].legend()
            axs[i,1].set_xlim(0,1)
            axs[i,1].set_yscale('symlog',linthresh=1e-4)
            axs[i,1].set_ylim(0,1)

            plt.tight_layout()

        if fig_savepath is not None:
            plt.savefig(fig_savepath)
            plt.close()
            print(f"Validation Plots Saved to '{fig_savepath}'")
            return None
        else:
            return fig, axs, bins


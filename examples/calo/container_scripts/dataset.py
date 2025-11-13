"""
Dataset for the Reconstruction model. Based on pytorch
"""
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ReconstructionDataset(Dataset):
    def __init__(
        self,
        input_df: pd.DataFrame,
        means: Optional[List[np.float32]] = None,
        stds: Optional[List[np.float32]] = None
    ):
        """Convert the files from the simulation to simple lists.

        Args:
            input_df (pd.DataFrame): Must contain as first level columns:
                ["Parameters", "Inputs", "Targets", "Context"], the names of the further dimensions
                are ignored.

        Returns:
            torch.DataSet
        """
        self.df = input_df
        self.df = self.filter_infs_and_nans(self.df)
        self.inputs = self.df["Inputs"].to_numpy("float32")
        self.targets = self.df["Targets"].to_numpy("float32")

        self.shape = (
            self.inputs.shape[1],
            self.targets.shape[1],
        )
        if means is None:
            self.means = [
                np.mean(self.inputs, axis=0),
                np.mean(self.targets, axis=0),
            ]
        else:
            self.means = means

        if stds is None:
            self.stds = [
                np.std(self.inputs, axis=0) + 1e-10,
                np.std(self.targets, axis=0) + 1e-10,
            ]
        else:
            self.stds = stds

        self.inputs = (self.inputs - self.means[0]) / self.stds[0]
        self.targets = (self.targets - self.means[1]) / self.stds[1]

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.c_means = [torch.tensor(a).to(dev) for a in self.means]
        self.c_stds = [torch.tensor(a).to(dev) for a in self.stds]

    def filter_infs_and_nans(self, df: pd.DataFrame):
        '''
        Removes all events that contain infs or nans.
        '''
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(axis=0, ignore_index=True)
        return df
    
    def filter_empty_events(self, df: pd.DataFrame):
        df = df[df["Inputs"]["sensor_energy_0"] > 0.0]
        df = df.dropna(axis=0, ignore_index=True)
        return df

    def unnormalize_target(self, target: torch.Tensor):
        return target * self.c_stds[1] + self.c_means[1]

    def normalize_target(self, target: torch.Tensor):
        return (target - self.c_means[1]) / self.c_stds[1]

    def unnormalize_detector(self, detector: torch.Tensor):
        return detector * self.c_stds[0] + self.c_means[0]
    
    def normalize_detector(self, detector: torch.Tensor):
        return (detector - self.c_means[0]) / self.c_stds[0]
        
    def __len__(self) -> int:
        return len(self.inputs)
    
    def __getitem__(self, idx: int):
        return self.inputs[idx], self.targets[idx]

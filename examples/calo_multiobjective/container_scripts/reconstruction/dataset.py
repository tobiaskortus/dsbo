"""
Dataset for the Reconstruction model. Based on pytorch
"""
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ReconstructionDataset(Dataset):
    def __init__(
        self,
        targets: Tuple[str],
        input_data: dict,
        means: Optional[List[np.float32]] = None,
        stds: Optional[List[np.float32]] = None
    ):
        """Convert the files from the simulation to simple lists.

        Args:
            input_mf (pd.DataFrame): Must contain as first level columns:
                ["Parameters", "Inputs", "Targets", "Context"], the names of the further dimensions
                are ignored.

        Returns:
            torch.DataSet
        """
        self.input_data = input_data
        self.input_data = self.filter_infs_and_nans(self.input_data)
        self.input_data = self.filter_empty_events(self.input_data)


        self.targets = np.concatenate(
            [
                self.input_data['Targets'][key].reshape(-1,1)
                for key in targets
            ],
            axis = 1,
        ).astype(np.float32)
        

        ls  = self.input_data['Inputs']['sensor_layer'].astype(np.float32)
        zs  = self.input_data['Inputs']['sensor_z'].astype(np.float32)
        dxs = self.input_data['Inputs']['sensor_dx'].astype(np.float32)
        dys = self.input_data['Inputs']['sensor_dy'].astype(np.float32)
        dzs = self.input_data['Inputs']['sensor_dz'].astype(np.float32)
        Es  = self.input_data['Inputs']['sensor_energy'].astype(np.float32)

        self.inputs = []
        for idx in np.unique(ls):
            mask = ls==idx
            assert np.all(mask == mask[0])
            mask = mask[0]
            E = Es[:,mask].sum(axis=1).reshape(-1,1)
            z = zs[:,mask][:,:1]
            dx = dxs[:,mask][:,:1]
            dy = dys[:,mask][:,:1]
            dz = dzs[:,mask][:,:1]
            assert all(dx>0)
            assert all(dy>0)
            assert all(dz>0)
            self.inputs.extend([np.log(1+E),z,dx,dy,dz])

        self.inputs = np.concatenate(self.inputs,axis=1)

        self.shape = (
            self.inputs.shape[1],
            self.targets.shape[1],
        )
        if means is None:
            self.means = [
                self.inputs.mean(axis=0),
                self.targets.mean(axis=0),
            ]
        else:
            self.means = means

        if stds is None:
            self.stds = [
                self.inputs.std(axis=0) + 1e-10,
                self.targets.std(axis=0) + 1e-10,
            ]
        else:
            self.stds = stds

        self.inputs = (self.inputs - self.means[0]) / self.stds[0]
        self.targets = (self.targets - self.means[1]) / self.stds[1]

        dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.c_means = [torch.tensor(a).to(dev) for a in self.means]
        self.c_stds = [torch.tensor(a).to(dev) for a in self.stds]

    def filter_infs_and_nans(self, data):
        '''
        Removes all events that contain infs or nans.
        '''
        for superkey, submf in data.items():
            for key in submf.keys():
                data[superkey][key] = np.nan_to_num(submf[key],nan=0.,posinf=0.,neginf=0.)
        return data

    def filter_empty_events(self, data):
        idx = np.where(data['Inputs']['sensor_energy'].sum(axis=1)>0)[0]
        for superkey, submf in data.items():
            for key in submf.keys():
                data[superkey][key] = submf[key][idx]
        return data


    def unnormalize_target(self, target: torch.Tensor):
        return target * self.c_stds[1].to(target.device) + self.c_means[1].to(target.device)

    def normalize_target(self, target: torch.Tensor):
        return (target - self.c_means[1]) / self.c_stds[1]

    def unnormalize_detector(self, detector: torch.Tensor):
        raise NotImplementedError()
        return detector * self.c_stds[1] + self.c_means[1]

    def normalize_detector(self, detector: torch.Tensor):
        raise NotImplementedError()
        return (detector - self.c_means[1]) / self.c_stds[1]

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        return self.inputs[idx], self.targets[idx]

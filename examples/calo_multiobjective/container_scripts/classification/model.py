from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import ClassificationDataset


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        in_groups = self.get_groups(self.in_channels)
        out_groups = self.get_groups(self.out_channels)

        # Pre-activation block
        self.block = nn.Sequential(
            nn.GroupNorm(in_groups, in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size, dilation=dilation, padding="same"),
            nn.GroupNorm(out_groups, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, dilation=dilation, padding="same"),
        )

        # Skip projection if channels change
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

        self.final_relu = nn.ReLU()

    @staticmethod
    def get_groups(channels):
        if channels <= 2:
            groups = 1
        elif channels <= 4:
            groups = 2
        elif channels <= 8:
            groups = 4
        else:
            groups = 8
        return groups

    def forward(self, x):
        out = self.block(x)
        skip = self.skip(x)
        return self.final_relu(out + skip)


class CNNBlocks(nn.Module):
    def __init__(self, channels, output_dim, dense_net=False, pooling_dim=1, dropout=0.):
        super().__init__()
        self.dense_net = dense_net
        self.pooling_dim = pooling_dim

        self.blocks = nn.ModuleList(
            [
                ResidualBlock(channels, 4, kernel_size=3, dilation=1),
                ResidualBlock(4, 8, kernel_size=3, dilation=2),
                ResidualBlock(8, 16, kernel_size=5, dilation=1),
            ]
        )

        self.dropout = nn.Dropout(p=dropout)
        self.avg_aggr = nn.AdaptiveAvgPool2d((self.pooling_dim,self.pooling_dim))
        self.max_aggr = nn.AdaptiveMaxPool2d((self.pooling_dim,self.pooling_dim))

        self.layer = nn.Sequential(
            nn.Linear(self.out_channels,output_dim*2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim*2,output_dim),
        )

    @property
    def out_channels(self):
        if self.dense_net:
            channels = sum([
                block.out_channels
                for block in self.blocks
            ])
        else:
            channels = self.blocks[-1].out_channels
        return channels * self.pooling_dim**2 * 2

    def forward(self, x):
        hs = [x]
        for block in self.blocks:
            hs.append(block(hs[-1]))
        if self.dense_net:
            x = torch.cat(hs[1:],dim=1)
        else:
            x = hs[-1]
        x = self.dropout(x)
        x = torch.cat([self.avg_aggr(x),self.max_aggr(x)],dim=1)
        x = x.view(x.size(0), -1)
        x = self.layer(x)
        return x

class Classification(nn.Module):
    def __init__(
        self,
        num_input_features: int,
        num_target_features: int,
        initial_means: List[np.float32],
        initial_stds: List[np.float32],
        multiclass: bool = False,
        weight: Tuple[float] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the shape of the model.

        Args:
            num_parameters (int): The number of detector parameters (length of the list of simulation parameters).
            num_input_features (int): The number of input features (sensors and other simulation outputs).
            num_target_features (int): The number of target features (quantities which are representatives
                of the detector's capabilities).
        """
        super().__init__()

        self.n_input_features = num_input_features
        self.n_target_features = num_target_features
        self.means = initial_means
        self.stds = initial_stds
        self.multiclass = multiclass
        self.weight = torch.tensor(weight) if weight is not None else None
        if self.multiclass:
            print ('Model initialised for multiclassification (one label per event)')

        out_dim = 0

        self.input_layers = []
        for input_features in self.n_input_features:
            if len(input_features) == 3:
                blocks = CNNBlocks(
                    channels = input_features[0],
                    output_dim = 32,
                    dense_net = False,
                    pooling_dim = 5,
                    dropout = 0.3,
                )
                self.input_layers.append(blocks)
                out_dim += 32
            elif len(input_features) == 1:
                self.input_layers.append(
                    nn.Sequential(
                        nn.Linear(input_features[0],32),
                        nn.ReLU(),
                        nn.Linear(32,16),
                    )
                )
                out_dim += 16
            else:
                raise NotImplementedError
        self.input_layers = nn.ModuleList(self.input_layers)

        self.final_layers = nn.Sequential(
            nn.Linear(out_dim,32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, num_target_features),
        )
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=1e-4)
        self.device = torch.device(device)

        self.print_children_params()


    def forward(self, x) -> torch.Tensor:
        """ Concatenate the detector parameters and the input
        """
        assert len(x) == len(self.input_layers)
        x = torch.cat(
            [
                layer(ix)
                for ix, layer in zip(x,self.input_layers)
            ],
            dim = 1,
        )
        
        return self.final_layers(x)

    @staticmethod
    def loss(y: torch.Tensor, y_pred: torch.Tensor, multiclass: bool, weight: torch.Tensor = None) -> torch.Tensor:
        assert y_pred.shape == y.shape, f'y has shape {y.shape}, but y_pred has shape {y_pred.shape}'
        if weight is not None:
            assert len(weight) == y_pred.shape[1]
        if multiclass:
            loss = nn.CrossEntropyLoss(reduction='none',weight=weight.to(y_pred.device))(y_pred,y)
        else:
            loss = nn.BCEWithLogitsLoss(reduction='none')(y_pred,y)
            if loss.dim() == 2 and loss.shape[1] > 1:
                if weight is not None:
                    loss = loss * weight.to(loss.device)
                loss = loss.mean(dim=-1)
        return loss

    def train_model(
        self,
        train_dataset: ClassificationDataset,
        valid_dataset: ClassificationDataset,
        batch_size: int,
        n_epochs: int,
        lr: float,
        plotter = None,
        early_stopping = None,
    ):
        if early_stopping is not None and early_stopping.early_stop:
            return
        print(f"Classification Training: {lr=}, {batch_size=}")
        print (f"Multiclass {self.multiclass}")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size*10, shuffle=False, num_workers=8)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.to(self.device)

        for epoch in range(n_epochs):
            # Training #
            self.train()
            train_losses = torch.zeros(len(train_loader))
            #print("Dataset length", len(train_dataset))
            for batch_idx, (x, y) in enumerate(train_loader):
                x: List[torch.Tensor] = [ix.to(self.device) for ix in x]
                y: torch.Tensor = y.to(self.device)
                y_pred: torch.Tensor = self(x)
                loss = self.loss(y,y_pred,self.multiclass,self.weight).mean()
                train_losses[batch_idx] = loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # Validation #
            self.eval()
            valid_losses = torch.zeros(len(valid_loader))
            for batch_idx, (x, y) in enumerate(valid_loader):
                x: List[torch.Tensor] = [ix.to(self.device) for ix in x]
                y: torch.Tensor = y.to(self.device)
                y_pred: torch.Tensor = self(x)
                loss = self.loss(y,y_pred,self.multiclass,self.weight).mean()
                valid_losses[batch_idx] = loss.item()

            print(f"Class Epoch: {epoch:4d} - Loss: {train_losses.mean():8.3f} - Val loss {valid_losses.mean():8.3f}")
            if plotter is not None:
                plotter.add_value('Loss (training)',train_losses.mean())
                plotter.add_value('Loss (validation)',valid_losses.mean())
                plotter.add_value('lr',lr)
            if early_stopping is not None:
                early_stopping(valid_losses.mean(),self)
                if early_stopping.early_stop:
                    print ('Early stopping')
                    break

        self.eval()

    def apply_model_in_batches(
        self,
        dataset: ClassificationDataset,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """ Apply the model in batches this is necessary because the model is too large to apply it to the
        whole dataset at once. The model is applied to the dataset in batches and the results are concatenated
        (the batch size is a hyperparameter).
        """
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        results = torch.zeros((len(dataset),self.n_target_features))
        loss_array = torch.zeros(len(dataset))
        mean_loss = 0.

        self.to(self.device)
        self.eval()

        for batch_idx, (x, y) in enumerate(data_loader):
            x: List[torch.Tensor] = [ix.to(self.device) for ix in x]
            y: torch.Tensor = y.to(self.device)
            with torch.no_grad():
                y_pred: torch.Tensor = self(x)
            loss = self.loss(y,y_pred,self.multiclass,self.weight)
            mean_loss += loss.mean().item()

            results[batch_idx * batch_size: (batch_idx + 1) * batch_size] = y_pred.cpu()
            loss_array[batch_idx * batch_size: (batch_idx + 1) * batch_size] = loss.flatten().cpu()

        mean_loss /= len(data_loader)
        return results.numpy(), loss_array.numpy(), mean_loss

    def print_children_params(self):
        print("\nParameters in top-level submodules:")
        print("----------------------------------")
        total = 0
        for name, module in self.named_children():
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"{name:20s}: {params:,}")
            total += params
        print("----------------------------------")
        print(f"Total trainable params: {total:,}\n")


class Classification2(nn.Module):
    def __init__(
        self,
        num_input_features: int,
        num_target_features: int,
        initial_means: List[np.float32],
        initial_stds: List[np.float32],
        multiclass: bool = False,
        weight: Tuple[float] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize the shape of the model.

        Args:
            num_parameters (int): The number of detector parameters (length of the list of simulation parameters).
            num_input_features (int): The number of input features (sensors and other simulation outputs).
            num_target_features (int): The number of target features (quantities which are representatives
                of the detector's capabilities).
        """
        super().__init__()

        self.n_input_features = num_input_features
        self.n_target_features = num_target_features
        self.means = initial_means
        self.stds = initial_stds
        self.multiclass = multiclass
        self.weight = torch.tensor(weight) if weight is not None else None
        if self.multiclass:
            print ('Model initialised for multiclassification (one label per event)')

        out_dim = 0
        print(self.n_input_features)
        self.pre_layers = nn.Sequential(
            nn.Conv2d(len(self.n_input_features) * 2, 16, kernel_size=3, padding='same'),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding='same'),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=5)
        )

        self.pos_layers = nn.Sequential(
            nn.Linear(800, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, num_target_features)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=1e-4)
        self.device = torch.device(device)

        self.print_children_params()


    def forward(self, x) -> torch.Tensor:
        """ Concatenate the detector parameters and the input
        """
        #assert len(x) == len(self.input_layers)
        x = torch.cat(x, dim = 1)
        x = self.pre_layers(x)
        x = x.view(x.size(0), -1)
        x = self.pos_layers(x)
        return x

    @staticmethod
    def loss(y: torch.Tensor, y_pred: torch.Tensor, multiclass: bool, weight: torch.Tensor = None) -> torch.Tensor:
        assert y_pred.shape == y.shape, f'y has shape {y.shape}, but y_pred has shape {y_pred.shape}'
        if weight is not None:
            assert len(weight) == y_pred.shape[1]
        if multiclass:
            loss = nn.CrossEntropyLoss(reduction='none',weight=weight.to(y_pred.device))(y_pred,y)
        else:
            loss = nn.BCEWithLogitsLoss(reduction='none')(y_pred,y)
            if loss.dim() == 2 and loss.shape[1] > 1:
                if weight is not None:
                    loss = loss * weight.to(loss.device)
                loss = loss.mean(dim=-1)
        return loss

    def train_model(
        self,
        train_dataset: ClassificationDataset,
        valid_dataset: ClassificationDataset,
        batch_size: int,
        n_epochs: int,
        lr: float,
        plotter = None,
        early_stopping = None,
    ):
        if early_stopping is not None and early_stopping.early_stop:
            return
        print(f"Classification Training: {lr=}, {batch_size=}")
        print (f"Multiclass {self.multiclass}")
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size*10, shuffle=False, num_workers=8)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        self.to(self.device)

        for epoch in range(n_epochs):
            # Training #
            self.train()
            train_losses = torch.zeros(len(train_loader))
            #print("Dataset length", len(train_dataset))
            for batch_idx, (x, y) in enumerate(train_loader):
                x: List[torch.Tensor] = [ix.to(self.device) for ix in x]
                y: torch.Tensor = y.to(self.device)
                y_pred: torch.Tensor = self(x)
                loss = self.loss(y,y_pred,self.multiclass,self.weight).mean()
                train_losses[batch_idx] = loss.item()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # Validation #
            self.eval()
            valid_losses = torch.zeros(len(valid_loader))
            for batch_idx, (x, y) in enumerate(valid_loader):
                x: List[torch.Tensor] = [ix.to(self.device) for ix in x]
                y: torch.Tensor = y.to(self.device)
                y_pred: torch.Tensor = self(x)
                loss = self.loss(y,y_pred,self.multiclass,self.weight).mean()
                valid_losses[batch_idx] = loss.item()

            print(f"Class Epoch: {epoch:4d} - Loss: {train_losses.mean():8.3f} - Val loss {valid_losses.mean():8.3f}")
            if plotter is not None:
                plotter.add_value('Loss (training)',train_losses.mean())
                plotter.add_value('Loss (validation)',valid_losses.mean())
                plotter.add_value('lr',lr)
            if early_stopping is not None:
                early_stopping(valid_losses.mean(),self)
                if early_stopping.early_stop:
                    print ('Early stopping')
                    break

        self.eval()

    def apply_model_in_batches(
        self,
        dataset: ClassificationDataset,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """ Apply the model in batches this is necessary because the model is too large to apply it to the
        whole dataset at once. The model is applied to the dataset in batches and the results are concatenated
        (the batch size is a hyperparameter).
        """
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        results = torch.zeros((len(dataset),self.n_target_features))
        loss_array = torch.zeros(len(dataset))
        mean_loss = 0.

        self.to(self.device)
        self.eval()

        for batch_idx, (x, y) in enumerate(data_loader):
            x: List[torch.Tensor] = [ix.to(self.device) for ix in x]
            y: torch.Tensor = y.to(self.device)
            with torch.no_grad():
                y_pred: torch.Tensor = self(x)
            loss = self.loss(y,y_pred,self.multiclass,self.weight)
            mean_loss += loss.mean().item()

            results[batch_idx * batch_size: (batch_idx + 1) * batch_size] = y_pred.cpu()
            loss_array[batch_idx * batch_size: (batch_idx + 1) * batch_size] = loss.flatten().cpu()

        mean_loss /= len(data_loader)
        return results.numpy(), loss_array.numpy(), mean_loss

    def print_children_params(self):
        print("\nParameters in top-level submodules:")
        print("----------------------------------")
        total = 0
        for name, module in self.named_children():
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"{name:20s}: {params:,}")
            total += params
        print("----------------------------------")
        print(f"Total trainable params: {total:,}\n")


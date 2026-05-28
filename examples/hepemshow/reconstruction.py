from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader

from dsbo.autograd import train_op
from dsbo.modules import dynamic_submodule
from dsbo.utils.io.autograd import NumpyFile
from dsbo.modules.reconstruction import ReconstructionBase
from dsbo.utils.io.workdir import Workdir, DefaultPaths, Input



def load_dataset(X_file: str, y_file: str, batch_size: int = 64, shuffle: bool = True,
                  aggr: tuple[torch.Tensor, torch.Tensor] | None = None,
                  detach_X: bool = True):

    X = NumpyFile.open(X_file, "cpu").float()
    y = NumpyFile.open(y_file, "cpu").float().detach()

    if detach_X:
        X = X.detach()

    if aggr is None:
        e_mean, e_std = y[:, 1].mean(), y[:,1].std()
        aggr = (e_mean, e_std)
    else:
        e_mean, e_std = aggr

    y[:,1] = (y[:,1] - e_mean) / e_std
    data_loader = DataLoader(TensorDataset(X, y), batch_size, shuffle)

    return data_loader, aggr


class ReconstructioModel(nn.Module):
    def __init__(self, num_detector_layers: int):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(num_detector_layers, 128), nn.LeakyReLU(),
                                    nn.Linear(128, 128), nn.LeakyReLU(),
                                    nn.Linear(128, 128))
        
        self.energy_estim = nn.Sequential(nn.Linear(128, 64), nn.LeakyReLU(),
                                          nn.Linear(64, 1))
        
        self.particle_id = nn.Sequential(nn.Linear(128, 64), nn.LeakyReLU(),
                                         nn.Linear(64, 3))
        

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.shared(x)
        return (self.particle_id(x), 
                self.energy_estim(x))
        


def train_model(model: nn.Module, train_X_file: str, train_y_file: str) -> nn.Module:
    optimizer = torch.optim.AdamW(model.parameters())
    train_losses = []
    val_losses = []

    train_loader, aggr = load_dataset(train_X_file, train_y_file)
    val_loader, aggr = load_dataset(train_X_file, train_y_file, batch_size=512, shuffle=False, aggr=aggr)

    def validate(val_loader: DataLoader) -> float:
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                y_particle = y[:, 0]
                y_energy = y[:, 1]
                y_hat_particle, y_hat_energy = model(X)
                particle_loss = F.cross_entropy(y_hat_particle, y_particle.long())
                energy_loss = F.mse_loss(y_hat_energy.squeeze(), y_energy)
                loss = particle_loss + energy_loss  
                val_loss += loss.item()

        model.train()
        return val_loss/len(val_loader)


    for epoch in tqdm(range(10)):
        for X, y in train_loader:
            y_particle = y[:, 0]
            y_energy = y[:, 1]
            y_hat_particle, y_hat_energy = model(X)

            particle_loss = F.cross_entropy(y_hat_particle, y_particle.long())
            energy_loss = F.mse_loss(y_hat_energy.squeeze(), y_energy)
            loss = particle_loss + energy_loss
            
            loss.backward()
            train_losses += [loss.item(), particle_loss.item(), energy_loss.item()]

            optimizer.step()
            optimizer.zero_grad()

        val_loss = validate(val_loader)
        val_losses += [val_loss]
        print(f"Epoch {epoch}: train loss = {train_losses[-3]:.4f}, particle loss = {train_losses[-2]:.4f}, energy loss = {train_losses[-1]:.4f}, val loss = {val_loss:.4f}")

    return model, aggr


def validate_model(
    model: nn.Module,
    val_X_file: str,
    val_y_file: str,
    aggr: tuple[torch.Tensor, torch.Tensor],
) -> dict[str, torch.Tensor]:
    val_set, _ = load_dataset(val_X_file, val_y_file,
                              batch_size=512, shuffle=False,
                              aggr=aggr, detach_X=False)
    p_losses = []
    e_losses = []

    for X, y in val_set:
        y_hat_p, y_hat_e = model(X)
        p_losses += [F.cross_entropy(y_hat_p, y[:,0].long(), reduction="none")]
        e_losses += [F.mse_loss(y_hat_e.squeeze(), y[:,1], reduction="none")]

    return {"crossentropy": torch.cat(p_losses),
            "mse": torch.cat(e_losses)}


class HepEmShowReconstruction(ReconstructionBase):
    def __init__(self, num_detector_layers: int = 50):
        super().__init__()
        self.num_detector_layers = num_detector_layers

        self.simulation_file_train = Input(Workdir.register_file(DefaultPaths.SIMULATION, "train_X.npz"))
        self.target_file_train = Input(Workdir.register_file(DefaultPaths.SIMULATION, "train_y.npz"))

        self.simulation_file_val = Input(Workdir.register_file(DefaultPaths.SIMULATION, "val_X.npz"))
        self.target_file_val = Input(Workdir.register_file(DefaultPaths.SIMULATION, "val_y.npz"))

        self.model_file = Workdir.register_file(DefaultPaths.RECONSTRUCTION, "model.pt")
        self.aggr_file = Workdir.register_file(DefaultPaths.RECONSTRUCTION, "normalization.pt")

    @train_op
    @dynamic_submodule
    def train(self, design, **kwargs):
        model = ReconstructioModel(self.num_detector_layers)
        model, aggr = train_model(model, self.simulation_file_train(), self.target_file_train())

        torch.save(model.state_dict(), self.model_file())
        torch.save(aggr, self.aggr_file())
    
    @dynamic_submodule
    def validate(self, design, **kwargs) -> dict[str, torch.Tensor]:
        model = ReconstructioModel(self.num_detector_layers)
        model.load_state_dict(torch.load(self.model_file()))

        aggr = torch.load(self.aggr_file())

        return validate_model(model,
                              self.simulation_file_val(), 
                              self.target_file_val(), 
                              aggr)
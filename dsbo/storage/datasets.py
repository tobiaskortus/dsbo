import torch
import lancedb
import numpy as np
import pyarrow as pa
from typing import Union
from lancedb.permutation import Permutation
from torch.utils.data import Dataset, DataLoader


def to_numpy(tensor: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor

def get_collate_fn(keys: list[str]):
    def collate_fn(batch: list[dict[str, Union[torch.Tensor, np.ndarray]]]) -> dict[str, np.ndarray]:
        return [torch.tensor([item[key] for item in batch]) for key in keys]
    return collate_fn


class SimulationDataset:
    def __init__(self, c: int, w: int, h: int, design_dim: int, params_dim: int):
        self._c, self._w, self._h = c, w, h
        self._design_dim = design_dim
        self._params_dim = params_dim

        self._schema = pa.schema([
            pa.field("design", pa.list_(pa.float32(), design_dim)),
            pa.field("params", pa.list_(pa.float32(), params_dim)),
            pa.field("simulation", pa.list_(pa.float32(), self._c * self._w * self._h))
        ])

        self._shapes = {
            "design": (design_dim,),
            "params": (params_dim,),
            "simulation": (self._c, self._w, self._h)
        }

        self._db = lancedb.connect("test_database")
        self._table = self._db.create_table("iteration_0", schema=self._schema, mode="overwrite")

    def _get_collate_fn(self, keys: list[str]):
        def collate_fn(batch: list[dict[str, Union[torch.Tensor, np.ndarray]]]) -> dict[str, np.ndarray]:
            return [torch.tensor([item[key] for item in batch]).reshape(-1, *self._shapes[key]) for key in keys]
        return collate_fn

    def write_batch(self, simulation: torch.Tensor | np.ndarray,
                          design: torch.Tensor | np.ndarray,
                          params: torch.Tensor | np.ndarray):
        
        simulation = to_numpy(simulation)
        design     = to_numpy(design)
        params     = to_numpy(params)

        n, m = simulation.shape[0], design.shape[0]
        assert m == n, "Batch size of simulation and design must match"

        design_dim = design.shape[1]
        params_dim = params.shape[1]
        img_flatten_dim = int(np.prod(simulation.shape[1:]))

        assert design_dim == self._design_dim, "Design shape does not match expected dimensions"
        assert params_dim == self._params_dim, "Params shape does not match expected dimensions"
        assert img_flatten_dim == self._c * self._w * self._h, "Simulation shape does not match expected dimensions"

        design_values = pa.array(design.reshape(-1), type=pa.float32())
        params_values = pa.array(params.reshape(-1), type=pa.float32())
        simulation_values = pa.array(simulation.reshape(-1), type=pa.float32())

        design_list = pa.FixedSizeListArray.from_arrays(design_values, design_dim)
        params_list = pa.FixedSizeListArray.from_arrays(params_values, params_dim)
        simulation_list = pa.FixedSizeListArray.from_arrays(simulation_values, img_flatten_dim)

        rows = pa.Table.from_arrays([design_list, params_list, simulation_list], names=["design", "params", "simulation"])
        self._table.add(rows)

    def cleanup(self):
        self._db.drop_table("iteration_0")

    @property
    def data_loader(self):
        collate_fn = self._get_collate_fn(["design", "params", "simulation"])
        return DataLoader(Permutation.identity(self._table), batch_size=64, collate_fn=collate_fn)
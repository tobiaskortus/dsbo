from abc import abstractmethod
import os
import numpy as np
import pandas as pd
from pathlib import Path

from dsbo.modules import Module
from dsbo.utils.io import Workdir

READERS = {
    ".csv":     pd.read_csv,
    ".json":    pd.read_json,
    ".xls":     pd.read_excel,
    ".xlsx":    pd.read_excel,
    ".parquet": pd.read_parquet,
    ".feather": pd.read_feather,
    ".pkl":     pd.read_pickle,
    ".pickle":  pd.read_pickle,
    ".npy": lambda file: pd.DataFrame(np.load(file, allow_pickle=True))
}

WRITERS = {
    ".csv":     pd.DataFrame.to_csv,
    ".json":    pd.DataFrame.to_json,
    ".xls":     pd.DataFrame.to_excel,
    ".xlsx":    pd.DataFrame.to_excel,
    ".parquet": pd.DataFrame.to_parquet,
    ".feather": pd.DataFrame.to_feather,
    ".pkl":     pd.DataFrame.to_pickle,
    ".pickle":  pd.DataFrame.to_pickle,
    ".npy": lambda df, file: np.save(file, df.to_records(index=False))
}


def load_df_by_extension(path: str, **kwargs):
    ext = os.path.splitext(path)[1].lower()
    if ext not in READERS:
        raise ValueError(f"Unsupported extension: {ext}")
    return READERS[ext](path, **kwargs), ext

def save_df_by_extension(df: pd.DataFrame, path: str, **kwargs):
    ext = os.path.splitext(path)[1].lower()
    if ext not in WRITERS:
        raise ValueError(f"Unsupported extension: {ext}")
    WRITERS[ext](df, path, **kwargs)


class SimulationTransform(Module):
    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, design: dict, **kwargs) -> None:
        return self.transform(design, **kwargs)

    @abstractmethod
    def transform(self, design: dict, **kwargs) -> None:
        pass


class DataFrameMergeTransform(SimulationTransform):
    def __init__(self, pattern: str) -> None:
        super().__init__()
        self.pattern = pattern
        self.simulation_dir = Workdir.register("simulation")
        self.output_file = Workdir.register_file("simulation", "output_df.npy", unique_filename=False)

    def transform(self, design, **kwargs):
        simulation_files = Path(self.simulation_dir()).glob(self.pattern)
        simulation_dfs = []
        for simulation_file in simulation_files:
            df, ext = load_df_by_extension(simulation_file)
            simulation_dfs.append(df)
            os.remove(simulation_file)

        assert len(simulation_dfs) > 0, "No simulation files found to merge."
            
        df = pd.concat(simulation_dfs, ignore_index=True)
        output_path = self.output_file()
        save_df_by_extension(df, output_path)

        
class MiniframeMergeTransform(SimulationTransform):
    def __init__(self, pattern: str) -> None:
        super().__init__()
        self.pattern = pattern
        self.simulation_files = Workdir.register_file_pattern("simulation", pattern)
        self.output_file = Workdir.register_file("simulation", "output_df.npy", unique_filename=False)

    def transform(self, design: dict) -> str:
        from dsbo.minipandas import MiniFrame, concat

        simulation_mfs = []
        for simulation_file in self.simulation_files():
            print(f"Loading simulation file: {simulation_file}")
            mf = MiniFrame.read_pickle(simulation_file)
            simulation_mfs.append(mf)
            os.remove(simulation_file)

        assert len(simulation_mfs) > 0, "No simulation files found to merge."
            
        merged_mf = concat(simulation_mfs, ignore_index=True)
        merged_mf.to_pickle(self.output_file())
import os
import numpy as np
import pandas as pd
from pathlib import Path

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


class SimulationTransform:
    def __init__(self) -> None:
        pass

    def __call__(self, simulation_dir: str) -> None:
        pass


class DataFrameMergeTransform(SimulationTransform):
    def __init__(self, pattern: str) -> None:
        super().__init__()
        self.pattern = pattern

    def __call__(self, simulation_dir: str) -> str:
        simulation_files = Path(simulation_dir).glob(self.pattern)
        simulation_dfs = []
        for simulation_file in simulation_files:
            df, ext = load_df_by_extension(simulation_file)
            simulation_dfs.append(df)

        assert len(simulation_dfs) > 0, "No simulation files found to merge."
            
        df = pd.concat(simulation_dfs, ignore_index=True)
        output_path = os.path.join(simulation_dir, "output_df.npy")
        save_df_by_extension(df, output_path)
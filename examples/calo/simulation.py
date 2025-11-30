import os
import json
import numpy as np
import pandas as pd
from typing import Optional, List

from dsbo.utils import python_script_args
from dsbo.simulation import SimulationContainerWrapperBase
from dsbo.transforms import SimulationTransform, load_df_by_extension

class G4CaloSimulation(SimulationContainerWrapperBase):
    def __init__(self, workdir: str, container_path: str, container_extra_flags: Optional[str] = ""):
        super().__init__(workdir, container_path, container_extra_flags)

    def simulate(self, design: dict, suffix: Optional[str] = None) -> None:
        suffix_str = f"_{suffix}" if suffix else ""
        # Make design file unique per worker to avoid race conditions
        design_path = os.path.join(self._workdir, f"design{suffix_str}.json")
        simulation_path = os.path.join(self.simulation_dir, f"output_df{suffix_str}.npy")

        with open(design_path, "w") as f:
            json.dump(design, f)

        args = python_script_args("examples/calo/container_scripts/simulation.py", [design_path, simulation_path])
        super().call_container(args, True)
        

class G4CaloSimToReco(SimulationTransform):
    def __init__(self) -> None:
        super().__init__()
        self.input_keys= ["sensor_energy", "sensor_x", "sensor_y", "sensor_z",
                          "sensor_dx", "sensor_dy", "sensor_dz", "sensor_layer"]
        self.target_keys= ["true_energy"]
        self.context_keys = ["true_pid"]  # Add any context features

    def __call__(self, simulation_dir: str) -> None:
        simulation_path = os.path.join(simulation_dir, "output_df.npy")
        output_path = os.path.join(simulation_dir, "output_df.parquet")
        def expand_columns(df: pd.DataFrame) -> pd.DataFrame:
            for column in df.columns:
                item = df[column][0]

                if isinstance(item, (list, np.ndarray)) and not isinstance(item, str):
                    column_list = df[column].tolist()
                    expanded_df = pd.DataFrame(column_list, index=df.index)
                    expanded_df.columns = [f"{column}_{i}" for i in range(len(expanded_df.columns))]
                    df = pd.concat([df.drop(columns=column), expanded_df], axis=1)

            return df
        
        df, ext = load_df_by_extension(simulation_path)
        
        # Create empty Parameters dataframe (no design parameters in simulation output)
        parameters_df = pd.DataFrame(index=df.index)
        
        df_combined_dict = {
                "Parameters": parameters_df,
                "Inputs": expand_columns(df[self.input_keys]),
                "Targets": expand_columns(df[self.target_keys]),
                "Context": expand_columns(df[self.context_keys])}

        df: pd.DataFrame = pd.concat(
                    df_combined_dict.values(),
                    keys=df_combined_dict.keys(),
                    axis=1)
        
        # Named levels are required for PyArrow to preserve MultiIndex
        df.columns.names = ['level_0', 'level_1']
        df = df.fillna(0)
        df = df.reset_index(drop=True)
        df.to_parquet(output_path, index=False, engine='fastparquet')
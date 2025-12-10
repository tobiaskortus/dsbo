from dataclasses import dataclass
from typing import Sequence, Union

@dataclass
class Objective:
    name: str
    minimize: bool = True

def check_objective_vals(objective_vals: Sequence[float], objectives: Union[Objective, Sequence[Objective]]) -> None:
    if isinstance(objectives, Objective):
        objectives = [objectives]
        
    if len(objective_vals) != len(objectives):
        raise ValueError(f"Number of objective values ({len(objective_vals)}) does not match number of objectives ({len(objectives)}).")


def is_single_objective(objective: Union[Objective, Sequence[Objective]]) -> bool:
    return isinstance(objective, Objective)

def is_multi_objective(objective: Union[Objective, Sequence[Objective]]) -> bool:
    return isinstance(objective, Sequence) and all(isinstance(obj, Objective) for obj in objective) and len(objective) > 1
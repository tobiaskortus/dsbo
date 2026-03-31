from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence, Union, TYPE_CHECKING

from dsbo.modules.compile import dry_run

if TYPE_CHECKING:
    from dsbo.workflow import Workflow

@dataclass
class Objective:
    name: str
    minimize: bool = True

def check_workflow_validity(workflow: Workflow, objectives: Union[Objective, Iterable[Objective]]):
    with dry_run():
        returns = workflow.run({})

    objectives = objectives if isinstance(objectives, Iterable) else [objectives]
    objective_keys = [obj.name for obj in objectives]
    assert set(returns.keys()).issuperset(set(objective_keys)), \
        f"Workflow did not return all objective values. Expected keys: {objective_keys}, got {list(returns.keys())}"

def get_objective_vals(returns: dict, objectives: list[Objective]) -> tuple[float]:
    return tuple([returns[obj.name] for obj in objectives])

def is_single_objective(objective: Union[Objective, Iterable[Objective]]) -> bool:
    return isinstance(objective, Objective)

def is_multi_objective(objective: Union[Objective, Iterable[Objective]]) -> bool:
    return isinstance(objective, Iterable) and all(isinstance(obj, Objective) for obj in objective) and len(objective) > 1
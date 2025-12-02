import optuna
from typing import Any, Optional, Callable, Sequence

from dsbo.workflow import Workflow
from dsbo.parameters import ValueRegistry
from dsbo.constraints import GeometryConstraintBase
from dsbo.parameters import DesignParameter, CompositeDesignParameter, ContinuousDesignParameter, CategoricalDesignParameter


def get_optuna_sampler_callback(trial: optuna.Trial) -> Callable[[DesignParameter], Any]:
    def callback(param: DesignParameter) -> Any:
        match param:
            case ContinuousDesignParameter():
                return trial.suggest_float(param.name, param.min_value, param.max_value)
            case CategoricalDesignParameter():
                return trial.suggest_categorical(param.name, param.choices)
            case _:
                if not issubclass(type(param), CompositeDesignParameter):
                    raise ValueError(f"Unsupported parameter type: {type(param)}")
    
    return callback


class OptunaBackend:
    def __init__(self, study_name: str, workflow: Workflow, design: CompositeDesignParameter, 
                 constraint: Optional[GeometryConstraintBase], seed: int = 42) -> None:
        self.design = design
        self.workflow = workflow
        self.constraint = constraint
        self.study_name = study_name
        constraint_fn = self.constraint_fn if self.constraint is not None else None
        self.sampler = optuna.samplers.TPESampler(constraints_func=constraint_fn, seed=seed)

    def sample_and_evaluate(self, trial: optuna.trial.Trial) -> float:
        sampler_callback = get_optuna_sampler_callback(trial)
        sampled_design = ValueRegistry.sample_from(self.design, sampler_callback)

        objective_val = self.workflow.run(sampled_design.to_dict(flatten=True))
        
        if self.constraint is not None:
            trial.set_user_attr("constraint", self.constraint(sampled_design.to_dict(flatten=True)))

        return objective_val[0]
    
    def constraint_fn(self, trial: optuna.trial.FrozenTrial) -> Sequence[float]:
        return trial.user_attrs["constraint"]

    def optimize(self, n_trials: int = 10):
        storage = f"sqlite:///{self.study_name}.db"
        study = optuna.create_study(study_name=self.study_name, storage=storage, sampler=self.sampler, 
                                    load_if_exists=False,  directions=["minimize"])

        study.optimize(self.sample_and_evaluate, n_trials=n_trials)
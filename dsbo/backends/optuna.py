import optuna
from optuna.storages import JournalStorage
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.storages.journal import JournalFileBackend
from typing import Any, Optional, Callable, Sequence, Union

from dsbo.workflow import Workflow
from dsbo.worker import DaskSlurmWorker
from dsbo.parameters import ValueRegistry
from dsbo.constraints import GeometryConstraintBase
from dsbo.objectives import Objective, check_objective_vals, is_single_objective
from dsbo.parameters import DesignParameter, CompositeDesignParameter, ContinuousDesignParameter, CategoricalDesignParameter


def get_optuna_sampler_callback(trial: optuna.Trial, log: bool = False) -> Callable[[DesignParameter], Any]:
    def callback(param: DesignParameter) -> Any:
        match param:
            case ContinuousDesignParameter():
                return trial.suggest_float(param.name, param.min_value, param.max_value, log=log)
            case CategoricalDesignParameter():
                return trial.suggest_categorical(param.name, param.choices)
            case _:
                if not issubclass(type(param), CompositeDesignParameter):
                    raise ValueError(f"Unsupported parameter type: {type(param)}")
    
    return callback

def get_directions(objective: Union[Objective, Sequence[Objective]]) -> Sequence[str]:
    if isinstance(objective, Objective):
        return ["minimize" if objective.minimize else "maximize"]
    else:
        return ["minimize" if obj.minimize else "maximize" for obj in objective]

class TrialFailedError(Exception):
    pass


class OptunaBackend:
    def __init__(self, study_name: str, workflow: Workflow, design: CompositeDesignParameter, 
                 constraint: Optional[GeometryConstraintBase], objective: Union[Objective, Sequence[Objective]],
                 worker: Optional[DaskSlurmWorker] = None,
                 seed: int = 42) -> None:
        
        self.seed = seed
        self.design = design
        self.worker = worker
        self.workflow = workflow
        self.objective = objective
        self.constraint = constraint
        self.study_name = study_name
        constraint_fn = self.constraint_fn if self.constraint is not None else None
        #self.sampler = optuna.samplers.TPESampler(n_startup_trials=30, multivariate=True, constant_liar=True,
        #                                          constraints_func=constraint_fn, seed=seed)
        #self.sampler = optuna.samplers.GPSampler(n_startup_trials=50, constraints_func=constraint_fn, seed=seed)
        self.sampler = optuna.integration.BoTorchSampler(n_startup_trials=50, constraints_func=constraint_fn, consider_running_trials=True, seed=seed)

    def sample_and_evaluate(self, trial: optuna.trial.Trial) -> Union[float, Sequence[float]]:
        sampler_callback = get_optuna_sampler_callback(trial, log=True)
        sampled_design = ValueRegistry.sample_from(self.design, sampler_callback)
        
        try:
            objective_val = self.workflow.run(sampled_design.to_dict(flatten=True))\
                  if self.worker is None\
                      else self.worker.submit(self.workflow, sampled_design.to_dict(flatten=True), trial.number).result()
            
            #Assert that returned objective values matches the expected objective structure
            check_objective_vals(objective_val, self.objective)
            
            if self.constraint is not None:
                trial.set_user_attr(_CONSTRAINTS_KEY, self.constraint(sampled_design.to_dict(flatten=True)))

            return objective_val[0] if is_single_objective(self.objective) else objective_val
        except Exception as e:
            print(f"Trial {trial.number} failed with error: {e}")
            raise TrialFailedError()
    
    def constraint_fn(self, trial: optuna.trial.FrozenTrial) -> Sequence[float]:
        return trial.user_attrs[_CONSTRAINTS_KEY]

    def optimize(self, n_trials: int = 10):
        #storage = f"sqlite:///{self.study_name}_{self.seed}.db"
        storage = JournalStorage(JournalFileBackend(f"{self.study_name}_{self.seed}.log"))
        study = optuna.create_study(study_name=f"{self.study_name}_{self.seed}", storage=storage, sampler=self.sampler, 
                                    load_if_exists=False, directions=get_directions(self.objective),)
    
        n_jobs = self.worker.n_worker if self.worker is not None else 1
        study.optimize(self.sample_and_evaluate, n_trials=n_trials, catch=(TrialFailedError,), n_jobs=n_jobs, gc_after_trial=True)
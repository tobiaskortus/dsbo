import os
import optuna
from contextlib import nullcontext
from optuna.storages import JournalStorage
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.storages.journal import JournalFileBackend
from typing import Any, Optional, Callable, Sequence, Union

from dsbo.utils.io import Workdir
from dsbo.worker.dask import AdaptiveHeterogenousDaskSlurmWorker, DaskSlurmWorker, WorkerContext
from dsbo.workflow import  WorkflowTaskGraph
from dsbo.parameters import ValueRegistry
from dsbo.constraints import GeometryConstraintBase
from dsbo.objectives import Objective, check_objective_vals, is_single_objective
from dsbo.parameters import DesignParameter, CompositeDesignParameter, ContinuousDesignParameter, CategoricalDesignParameter

class TrialFailedError(Exception):
    """Custom exception to indicate a trial failure in Optuna optimization."""
    pass

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


class OptunaBackend:
    def __init__(self, study_name: str, workflow: WorkflowTaskGraph, design: CompositeDesignParameter, 
                 constraint: Optional[GeometryConstraintBase], objective: Union[Objective, Sequence[Objective]],
                 worker: Optional[Union[DaskSlurmWorker, AdaptiveHeterogenousDaskSlurmWorker]] = None,
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
        self.sampler = optuna.integration.BoTorchSampler(n_startup_trials=100, constraints_func=constraint_fn, consider_running_trials=True, seed=seed)
        #self.sampler = optuna.samplers.RandomSampler(seed=seed)

    def sample_and_evaluate(self, trial: optuna.trial.Trial) -> Union[float, Sequence[float]]:
        with WorkerContext.use_worker(self.worker) if self.worker is not None else nullcontext():
            sampler_callback = get_optuna_sampler_callback(trial, log=True)
            sampled_design = ValueRegistry.sample_from(self.design, sampler_callback)
            
            try:
                workdir_root = os.path.join("files", f"run{self.seed}", f"trial_{trial.number}")
                #TODO: should create throw an error if workdir exists?
                with Workdir.create(workdir_root, error_if_exists=False) as wd:
                    objective_val = self.workflow.run(sampled_design.to_dict(flatten=True))\
                    
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
                                    load_if_exists=False, directions=get_directions(self.objective))
    
        #n_jobs = self.worker.n_worker if self.worker is not None else 1
        #TODO: set number of workers properly (considering adaptive workers)
        study.optimize(self.sample_and_evaluate, n_trials=n_trials, catch=(TrialFailedError,), n_jobs=20, gc_after_trial=True)
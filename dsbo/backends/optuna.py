import gc
import os
import torch
import optuna
import logging
from contextlib import nullcontext
from optuna.storages import JournalStorage
from optuna.samplers._base import _CONSTRAINTS_KEY
from optuna.storages.journal import JournalFileBackend
from typing import Any, Iterable, Optional, Callable, Sequence, Union

from dsbo.utils.io import Workdir
from dsbo.worker.dask import  WorkerContext, DaskWorkerBase
from dsbo.workflow import  WorkflowTaskGraph
from dsbo.parameters import Candidate, Continuous, Integer, Categorical, Sampler, ParameterSpace
from dsbo.constraints import GeometryConstraintBase
from dsbo.objectives import Objective, is_single_objective, check_workflow_validity, get_objective_vals
from dsbo.parameters import DesignParameter
from optuna_integration.botorch import qehvi_candidates_func

def qehvi_candidates_func_wrapper(
    train_x,
    train_obj,
    train_con,
    bounds,
    pending_x,
):
    import torch
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info("Generating candidates using QEHVI...")
    
    try:
        result = qehvi_candidates_func(
            train_x,
            train_obj,
            train_con,
            bounds,
            pending_x,
        )
        return result
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


class TrialFailedError(Exception):
    """Custom exception to indicate a trial failure in Optuna optimization."""
    pass

class OptunaSampler(Sampler):
    def __init__(self, trial: optuna.trial.Trial):
        self._trial = trial

    def ask(self) -> torch.Tensor:
        values = {}
        for name, p in self._space:
            match p:
                case Continuous():
                    values[name] = self._trial.suggest_float(name, p.lower, p.upper, log=p.log)
                case Integer():
                    values[name] = self._trial.suggest_int(name, p.lower, p.upper)
                case Categorical():
                    values[name] = self._trial.suggest_categorical(name, p.choices)
        return self._space.encode(values)


def get_directions(objective: Union[Objective, Sequence[Objective]]) -> Sequence[str]:
    if isinstance(objective, Objective):
        return ["minimize" if objective.minimize else "maximize"]
    else:
        return ["minimize" if obj.minimize else "maximize" for obj in objective]


class OptunaBackend:
    def __init__(self, study_name: str, workflow: WorkflowTaskGraph, design: ParameterSpace, 
                 constraint: Optional[GeometryConstraintBase], objective: Union[Objective, Iterable[Objective]],
                 worker: Optional[DaskWorkerBase] = None,
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.sampler = optuna.integration.BoTorchSampler(candidates_func=qehvi_candidates_func_wrapper, n_startup_trials=100, constraints_func=constraint_fn, 
                                                         consider_running_trials=True, seed=seed, device=self.device)
        #self.sampler = optuna.samplers.RandomSampler(seed=seed)

        check_workflow_validity(self.workflow, self.objective)

    def sample_design(self, trial: optuna.trial.Trial) -> Candidate:
        """Only sample the design parameters for a trial (this triggers GP fitting in the sampler)."""
        return self.design.sample(OptunaSampler(trial))

    def evaluate_design(self, trial_number: int, candidate: Candidate) -> tuple[Union[float, Sequence[float]], Optional[Sequence[float]]]:
        """
        Evaluate the sampled design (run workflow, check objectives, set constraints).
        Returns (objective_values, constraints)
        """
        with WorkerContext.use_worker(self.worker) if self.worker is not None else nullcontext():
            try:
                workdir_root = os.path.join("files", f"run{self.seed}", f"trial_{trial_number}")
                with Workdir.create(workdir_root):
                    result = self.workflow.run(candidate.values)
                    objective_val = get_objective_vals(result, self.objective)
                    
                    constraints = None
                    if self.constraint is not None:
                        constraints = self.constraint(candidate.values)
                    
                    return objective_val[0] if is_single_objective(self.objective) else objective_val, constraints
                
            except Exception as e:
                print(f"Trial {trial_number} failed with error: {e}")                
                raise TrialFailedError()
            
    def sample_and_evaluate(self, trial: optuna.trial.Trial) -> Union[float, Sequence[float]]:
        candidate = self.sample_design(trial)
        return self.evaluate_design(trial, candidate)
    
    def constraint_fn(self, trial: optuna.trial.FrozenTrial) -> Sequence[float]:
        return trial.user_attrs[_CONSTRAINTS_KEY]

    def optimize(self, n_trials: int = 10):
        storage = JournalStorage(JournalFileBackend(f"{self.study_name}_{self.seed}.log"))
        study = optuna.create_study(study_name=f"{self.study_name}_{self.seed}", storage=storage, sampler=self.sampler, 
                                    load_if_exists=False, directions=get_directions(self.objective))
    
        study.optimize(self.sample_and_evaluate, n_trials=n_trials, catch=(TrialFailedError,), n_jobs=20, gc_after_trial=True)
    
    def optimize_ask_tell(self, n_trials: int = 100, num_workers: int = 20, num_samplers: int = 1):
        import asyncio
        import multiprocessing as mp
        from queue import Empty
        
        def study_worker(ask_queue, result_queue, study_name, seed, sampler, objective, design):
            """Worker process that manages the study and handles ask/tell operations."""
            storage = JournalStorage(JournalFileBackend(f"{study_name}_{seed}.log"))
            study = optuna.create_study(
                study_name=f"{study_name}_{seed}",
                storage=storage,
                sampler=sampler,
                load_if_exists=False,
                directions=get_directions(objective)
            )
            
            while True:
                try:
                    msg = ask_queue.get()
                    
                    if msg['type'] == 'ask':
                        trial = study.ask()
                        trial_number = trial.number
                        
                        candidate = design.sample(OptunaSampler(trial))

                        del trial
                        gc.collect()
                        
                        result_queue.put({
                            'type': 'design',
                            'design': candidate,
                            'trial_number': trial_number
                        })
                    
                    elif msg['type'] == 'tell':
                        trial_number = msg['trial_number']
                        value = msg['value']
                        state = msg['state']
                        constraints = msg.get('constraints')
                        
                        #TODO: this is terrible !!!
                        if constraints is not None:
                            trials = [t for t in study.get_trials(deepcopy=False) if t.number == trial_number]
                            if trials:
                                study._storage.set_trial_user_attr(trials[0]._trial_id, _CONSTRAINTS_KEY, constraints)
                        
                        study.tell(trial_number, value, state)
                        
                        gc.collect()
                        
                        result_queue.put({'type': 'ack'})
                    
                    elif msg['type'] == 'shutdown':
                        break
                        
                except Exception as e:
                    result_queue.put({'type': 'error', 'error': str(e)})
        
        ask_queue = mp.Queue()
        result_queue = mp.Queue()
        worker_process = mp.Process(
            target=study_worker,
            args=(ask_queue, result_queue, self.study_name, self.seed, self.sampler, self.objective, self.design)
        )
        worker_process.start()
        
        trial_semaphore = asyncio.Semaphore(num_workers)
        ask_semaphore = asyncio.Semaphore(num_samplers)
        
        n_startup_trials = getattr(self.sampler, '_n_startup_trials')
        asked_count = [0]
        completed_count = [0]
        ask_condition = asyncio.Condition()
        
        async def run_trial():
            async with trial_semaphore:
                loop = asyncio.get_event_loop()
                
                async with ask_semaphore:
                    async with ask_condition:
                        if asked_count[0] >= n_startup_trials and completed_count[0] < n_startup_trials:
                            print(f"Trial waiting: asked={asked_count[0]}, completed={completed_count[0]}")
                        while asked_count[0] >= n_startup_trials and completed_count[0] < n_startup_trials:
                            await ask_condition.wait()
                        
                        ask_queue.put({'type': 'ask'})
                        response = await loop.run_in_executor(None, result_queue.get)
                        
                        if response['type'] == 'error':
                            print(f"Error getting trial: {response['error']}")
                            return
                        
                        trial_number = response['trial_number']
                        candidate = response['design']
                        asked_count[0] += 1
                        print(f"Asked trial {trial_number}, total asked: {asked_count[0]}")
            
                try:
                    value, constraints = await loop.run_in_executor(None, self.evaluate_design, trial_number, candidate)
                    
                    ask_queue.put({
                        'type': 'tell',
                        'trial_number': trial_number,
                        'value': value,
                        'state': None,
                        'constraints': constraints
                    })
                
                    await loop.run_in_executor(None, result_queue.get)
                    
                except TrialFailedError:
                    ask_queue.put({
                        'type': 'tell',
                        'trial_number': trial_number,
                        'value': None,
                        'state': optuna.trial.TrialState.FAIL,
                        'constraints': None
                    })
                    
                    await loop.run_in_executor(None, result_queue.get)
                    
                except Exception as e:
                    print(f"Trial failed with error: {e}")
                    ask_queue.put({
                        'type': 'tell',
                        'trial_number': trial_number,
                        'value': None,
                        'state': optuna.trial.TrialState.FAIL,
                        'constraints': None
                    })
                    
                    await loop.run_in_executor(None, result_queue.get)
                    
                finally:
                    del candidate
                    gc.collect()
                    
                    async with ask_condition:
                        completed_count[0] += 1
                        ask_condition.notify_all()

        async def main():
            tasks = [asyncio.create_task(run_trial()) for _ in range(n_trials)]
            await asyncio.gather(*tasks)
            
            ask_queue.put({'type': 'shutdown'})
            worker_process.join()

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return asyncio.create_task(main())
            else:
                loop.run_until_complete(main())
        except RuntimeError:
            asyncio.run(main())
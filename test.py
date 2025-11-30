import optuna
from copy import deepcopy
from typing import Any, Union, Optional
from dsbo.parameters import DesignParameter, ContinuousDesignParameter, CategoricalDesignParameter

from dsbo.constraints import ConstraintFunctionBase, MultiConstraint
from examples.calo.constraints import G4CaloCostConstraint, G4CaloLengthConstraint

from dsbo.workflow import Workflow
from dsbo.simulation import DaskSimulationWrapper   
from dsbo.transforms import DataFrameMergeTransform
from dsbo.parameters import ContinuousDesignParameter, CategoricalDesignParameter, CompositeDesignParameter

from examples.calo.simulation import G4CaloSimulation, G4CaloSimToReco
from examples.calo.reconstruction import G4CaloReconstruction


class CaloDesignParameterization(CompositeDesignParameter):
    def __init__(self, name: str) -> None:
        parameters = []
        for i in range(3):
            absorber = CompositeDesignParameter(
                name=f"absorber_{i}",
                parameters=[ContinuousDesignParameter(f"thickness_absorber_{i}", 0.0, 100.0),
                            CategoricalDesignParameter(f"material_absorber_{i}", ["G4_PbWO4", "G4_POLYSTYRENE"])]
            )
            scintilator = CompositeDesignParameter(
                name=f"scintilator_{i}",
                parameters=[ContinuousDesignParameter(f"thickness_scintillator_{i}", 0.0, 100.0),
                            CategoricalDesignParameter(f"material_scintillator_{i}", ["G4_Pb", "G4_Fe"])]
            )
            parameters += [absorber, scintilator]

        super().__init__(name, parameters)


calo_design = CaloDesignParameterization("calo")

def sample_parameters(param: Union[DesignParameter, CompositeDesignParameter], trial: optuna.trial.Trial):
    if isinstance(param, ContinuousDesignParameter):
        param.current_value = trial.suggest_float(param.name, param.min_value, param.max_value)
        return
    elif isinstance(param, CategoricalDesignParameter):
        param.current_value = trial.suggest_categorical(param.name, param.choices)
        return

    for p in param._parameters:
        sample_parameters(p, trial)

class OptunaBackend:
    def __init__(self, workflow: Workflow, design: CompositeDesignParameter, constraint: Optional[ConstraintFunctionBase]) -> None:
        self.constraint = constraint
        self.workflow = workflow
        self.design = design
        constraints_fn = self.constraint_fn if self.constraint is not None else None
        self.sampler = optuna.samplers.TPESampler(constraints_func=constraints_fn)

    def sample_design(self, trial: optuna.trial.Trial) -> CompositeDesignParameter:
        sampled_design = deepcopy(self.design)
        sample_parameters(sampled_design, trial)
        return sampled_design 

    def sample_and_evaluate(self, trial: optuna.trial.Trial) -> float:
        sampled_design = self.sample_design(trial)
        objective_val = self.workflow.run(sampled_design.to_dict(flatten=True))
        if self.constraint is not None:
            trial.set_user_attr("constraint", self.constraint(sampled_design))
        return objective_val[0]
    
    def constraint_fn(self, trial: optuna.trial.Trial) -> tuple[float]:
        return trial.user_attrs["constraint"]

    def optimize(self, n_trials: int = 10) -> Any:
        study_name = "example-study-calo"  # Unique identifier of the study.
        storage_name = "sqlite:///{}.db".format(study_name)
        study = optuna.create_study(study_name=study_name, storage=storage_name, sampler=self.sampler, 
                                    load_if_exists=False,  directions=["minimize"])

        study.optimize(self.sample_and_evaluate, n_trials=n_trials)


if __name__ == "__main__":
    transform = G4CaloSimToReco()
    base_simulation = G4CaloSimulation(workdir="files", container_path="/scratch/nyl93mev/aido/minicalosim_7829fde_2.sif")
    merge_transform = DataFrameMergeTransform("output_df_*")
    simulation = DaskSimulationWrapper(base_simulation=base_simulation, n_simulations=10, 
                                       n_workers=10, threads_per_worker=1,
                                       merge_function=merge_transform)
    reconstruction = G4CaloReconstruction(workdir="files", container_path="/scratch/nyl93mev/aido/minicalosim_7829fde_2.sif")
    constraint = MultiConstraint(G4CaloCostConstraint(max_cost=200_000),
                                 G4CaloLengthConstraint(max_length=200))

    workflow = Workflow(simulation=simulation,
                        postprocessing=transform,
                        reconstruction=reconstruction)

    OptunaBackend(workflow, calo_design, constraint).optimize(n_trials=500)
    simulation.close()
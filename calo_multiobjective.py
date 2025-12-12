from argparse import ArgumentParser
from dataclasses import dataclass
from typing import List

from dsbo.objectives import Objective
from examples.calo.constraints import G4CaloCostConstraint, G4CaloLengthConstraint
from examples.calo_multiobjective.reconstruction import G4CaloReconstruction
from examples.calo_multiobjective.simulation import G4CaloSimulation, G4CaloSimToReco
#from examples.calo_multitask.constraints import G4CaloCostConstraint, G4CaloLengthConstraint

from dsbo.workflow import Workflow
from dsbo.worker import DEFAULT_CPU_CONFIG, DaskSlurmWorker, DEFAULT_GPU_CONFIG
from dsbo.constraints import MultiConstraint
from dsbo.backends.optuna import OptunaBackend
from dsbo.simulation import DaskSimulationWrapper
from dsbo.transforms import DataFrameMergeTransform, MiniframeMergeTransform
from dsbo.parameters import ContinuousDesignParameter, CategoricalDesignParameter, CompositeDesignParameter, ValueRegistry
from dsbo.backends.rodeo import get_experiment_of_design_callback


class CaloDesignParameterization(CompositeDesignParameter):
    def __init__(self, name: str) -> None:
        parameters = []
        for i in range(5):
            absorber = CompositeDesignParameter(
                name=f"absorber_{i}",
                parameters=[ContinuousDesignParameter(f"thickness_absorber_{i}", 1e-3, 50.0),
                            CategoricalDesignParameter(f"material_absorber_{i}", ["G4_Pb", "G4_Fe"])]
            )
            scintillator = CompositeDesignParameter(
                name=f"scintilator_{i}",
                parameters=[ContinuousDesignParameter(f"thickness_scintillator_{i}", 1e-3, 50.0),
                            CategoricalDesignParameter(f"material_scintillator_{i}", ["G4_PbWO4", "G4_POLYSTYRENE"])]
            )
            parameters += [absorber, scintillator]

        super().__init__(name, parameters)



if __name__ == "__main__":
    seed = 3
    calo_design = CaloDesignParameterization("calo")

    transform = G4CaloSimToReco()
    base_simulation = G4CaloSimulation(workdir=f"files/run{seed}", container_path="/home/nyl93mev/minicalosim_90baa21.sif")
    merge_transform = MiniframeMergeTransform("output_df_*")
    simulation      = DaskSimulationWrapper(base_simulation=base_simulation, n_simulations=30, n_workers=10, 
                                            threads_per_worker=1, merge_function=merge_transform, init_on_demand=True)

    reconstruction = G4CaloReconstruction(workdir=f"files/run{seed}", container_path="/home/nyl93mev/minicalosim_90baa21.sif")
    
    constraint = MultiConstraint(G4CaloCostConstraint(max_cost=100_000),
                                 G4CaloLengthConstraint(max_length=200))

    workflow = Workflow(simulation=simulation,
                        postprocessing=transform,
                        reconstruction=reconstruction)
    
    objectives = [Objective(name="energy_resolution", minimize=True),
                  Objective(name="particle_id", minimize=True)]
    
    with DaskSlurmWorker(n_worker=10, worker_config=DEFAULT_GPU_CONFIG) as worker:
        OptunaBackend(f"calo_constrained", workflow, calo_design, constraint, objectives, worker, seed=seed).optimize(n_trials=500)


 



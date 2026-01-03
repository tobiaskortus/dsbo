from argparse import ArgumentParser
from dataclasses import dataclass

from dsbo.objectives import Objective
from dsbo.workflow import DefaultSimRecoWorkflow, FusedWorkflowTask, WorkflowTask, WorkflowTaskGraph, optimize, workflow_task
from examples.calo.constraints import G4CaloCostConstraint, G4CaloLengthConstraint
from examples.calo_multiobjective.reconstruction import G4CaloReconstruction
from examples.calo_multiobjective.simulation import G4CaloSimulation, G4CaloSimToReco
#from examples.calo_multitask.constraints import G4CaloCostConstraint, G4CaloLengthConstraint

from dsbo.worker.config import DEFAULT_CPU_CONFIG, DEFAULT_GPU_CONFIG
from dsbo.worker.dask import DaskSlurmWorker, AdaptiveHeterogenousDaskSlurmWorker
from dsbo.constraints import MultiConstraint
from dsbo.backends.optuna import OptunaBackend
from dsbo.modules.simulation import DaskSimulationWrapper
from dsbo.transforms import DataFrameMergeTransform, MiniframeMergeTransform
from dsbo.parameters import ContinuousDesignParameter, CategoricalDesignParameter, CompositeDesignParameter, ValueRegistry


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
    seed = 47
    calo_design = CaloDesignParameterization("calo")

    @workflow_task("simulation", DEFAULT_CPU_CONFIG)
    def create_simulation(n_simulations=30, n_workers=15):
        base_simulation = G4CaloSimulation(container_path="/home/nyl93mev/minicalosim_90baa21.sif")
        merge_transform = MiniframeMergeTransform("output_df_*")
        simulation      = DaskSimulationWrapper(base_simulation=base_simulation, n_simulations=n_simulations, n_workers=n_workers, 
                                                threads_per_worker=1, merge_function=merge_transform, init_on_demand=True)
        return simulation

    @workflow_task("sim2reco", DEFAULT_CPU_CONFIG)
    def create_sim2reco():
        return G4CaloSimToReco()

    @workflow_task("reconstruction", DEFAULT_GPU_CONFIG)
    def create_reconstruction():
        return G4CaloReconstruction(container_path="/home/nyl93mev/minicalosim_90baa21.sif")
    
    constraint = MultiConstraint(G4CaloCostConstraint(max_cost=100_000),
                                 G4CaloLengthConstraint(max_length=200))

    simulation = create_simulation()
    sim2reco  = create_sim2reco()
    reconstruction = create_reconstruction()

    workflow = DefaultSimRecoWorkflow(simulation, sim2reco, reconstruction)
    workflow = optimize(workflow)
    workflow.visualize("calo_multiobjective_workflow.png")
    
    objectives = [Objective(name="energy_resolution", minimize=True),
                  Objective(name="particle_id", minimize=True)]
    

    with AdaptiveHeterogenousDaskSlurmWorker(max_workers=10, logdir="logs") as worker:
        OptunaBackend(f"calo_constrained", workflow, calo_design, constraint, objectives, worker, seed=seed).optimize(n_trials=500)


 



from dsbo.objectives import Objective
from dsbo.constraints import MultiConstraint
from dsbo.backends.optuna import OptunaBackend
from dsbo.transforms import MiniframeMergeTransform
from dsbo.modules.simulation import DaskSimulationWrapper
from dsbo.worker.dask import AdaptiveHeterogeneousDaskSlurmWorker
from dsbo.worker.config import DEFAULT_CPU_CONFIG, DEFAULT_GPU_CONFIG, SlurmWorkerConfig
from dsbo.workflow import DefaultSimRecoWorkflow, WorkflowTaskGraph, optimize
from dsbo.parameters import ContinuousDesignParameter, CategoricalDesignParameter, CompositeDesignParameter

from examples.calo_multiobjective.reconstruction import G4CaloReconstruction
from examples.calo.constraints import G4CaloCostConstraint, G4CaloLengthConstraint
from examples.calo_multiobjective.simulation import G4CaloSimulation, G4CaloSimToReco

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
    seed = 43
    calo_design = CaloDesignParameterization("calo")
    constraint = MultiConstraint(G4CaloCostConstraint(max_cost=100_000),
                                 G4CaloLengthConstraint(max_length=200))

    
    SINGLE_CORE_CONFIG = SlurmWorkerConfig(processes=2, job_cpu=2, walltime="48:00:00", memory="16GB", partition="skylake-96")
    DEFAULT_GPU_CONFIG = SlurmWorkerConfig(job_cpu=4, gpu_resources="gpu:1", memory="16GB", walltime="48:00:00")

    simulation       = G4CaloSimulation("/home/nyl93mev/minicalosim_90baa21.sif").to(SINGLE_CORE_CONFIG)
    simulation_merge = MiniframeMergeTransform("output_df_*").to(SINGLE_CORE_CONFIG)
    sim2reco         = G4CaloSimToReco().to(SINGLE_CORE_CONFIG)
    reconstruction   = G4CaloReconstruction("/home/nyl93mev/minicalosim_90baa21.sif").to(DEFAULT_GPU_CONFIG)

    workflow = WorkflowTaskGraph(simulation * 10 >> simulation_merge >> sim2reco >> reconstruction)
    workflow = optimize(workflow)
    workflow.visualize("workflow_dsl_example.png")
    
    objectives = [Objective(name="energy_resolution", minimize=True),
                  Objective(name="particle_id", minimize=True)]
    
    with AdaptiveHeterogeneousDaskSlurmWorker.from_workflow(workflow, logdir="logdir") as worker:
        OptunaBackend(f"calo_constrained", workflow, calo_design, constraint, objectives, worker, seed=seed).optimize_ask_tell(n_trials=500)


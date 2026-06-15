from dsbo.objectives import Objective
from dsbo.constraints import MultiConstraint
from dsbo.backends.optuna import OptunaBackend
from dsbo.transforms import MiniframeMergeTransform
from dsbo.worker.dask import HeterogeneousDaskWorker
from dsbo.worker.spec import DEFAULT_CPU_CONFIG, DEFAULT_GPU_CONFIG, SLURMWorkerSpec, WorkerSpec
from dsbo.workflow import  Workflow, optimize
from dsbo.parameters import ParameterSpace, Group, Continuous, Categorical

from examples.calo_multiobjective.reconstruction import G4CaloReconstruction
from examples.calo_multiobjective.constraints import G4CaloCostConstraint, G4CaloLengthConstraint
from examples.calo_multiobjective.simulation import G4CaloSimulation, G4CaloSimToReco

def _g4_calo_key(parts):
        return f"{parts[-1]}_{parts[-2]}_{parts[0].split('_')[-1]}"

calorimeter = ParameterSpace(
    key_fn=_g4_calo_key,
    **ParameterSpace.repeat('layer', 5,
        absorber=Group(
            thickness=Continuous(0.1, 50.0, log=True),
            material=Categorical(['G4_Pb', 'G4_Fe']),
        ),
        scintillator=Group(
            thickness=Continuous(0.1, 50.0, log=True),
            material=Categorical(['G4_PbWO4', 'G4_POLYSTYRENE']),
        ),
    )
)


if __name__ == "__main__":
    seed = 43
    constraint = MultiConstraint(G4CaloCostConstraint(max_cost=100_000),
                                 G4CaloLengthConstraint(max_length=200))
    
    SINGLE_CORE_CONFIG = SLURMWorkerSpec(processes=5, cores=5, walltime="48:00:00", memory="48GB", partition="epyc-256")
    DEFAULT_GPU_CONFIG = SLURMWorkerSpec(cores=4, gpu_resources="gpu:1", memory="16GB", walltime="48:00:00")

    simulation       = G4CaloSimulation("/home/nyl93mev/minicalosim_90baa21.sif").to(SINGLE_CORE_CONFIG)
    simulation_merge = MiniframeMergeTransform("output_df_*").to(SINGLE_CORE_CONFIG)
    sim2reco         = G4CaloSimToReco().to(SINGLE_CORE_CONFIG)
    reconstruction   = G4CaloReconstruction("/home/nyl93mev/minicalosim_90baa21.sif").to(DEFAULT_GPU_CONFIG)

    workflow = Workflow(simulation * 30 >> simulation_merge >> sim2reco >> reconstruction)
    workflow = optimize(workflow)
    workflow.visualize("workflow_dsl_example.png")
    
    objectives = [Objective(name="energy_resolution", minimize=True),
                  Objective(name="particle_id", minimize=True)]
    
    with HeterogeneousDaskWorker.from_workflow(workflow, logdir="logdir") as worker:
        OptunaBackend(f"calo_constrained", workflow, calorimeter, constraint, objectives, worker, seed=seed).optimize_ask_tell(n_trials=500)


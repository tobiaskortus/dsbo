import torch
import numpy as np

from dsbo.objectives import Objective
from dsbo.constraints import MultiConstraint
from dsbo.backends.optuna import OptunaBackend
from dsbo.backends.aido import AidoBackend
from dsbo.transforms import MiniframeMergeTransform
from dsbo.worker.dask import AdaptiveHeterogeneousDaskSlurmWorker
from dsbo.worker.config import DEFAULT_CPU_CONFIG, DEFAULT_GPU_CONFIG, SlurmWorkerConfig
from dsbo.workflow import  Workflow, optimize
from dsbo.parameters import Constant, Constant, ParameterSpace, Group, Continuous, Categorical, Integer, Sampler

from examples.calo_multiobjective.reconstruction import G4CaloReconstruction
from examples.calo_multiobjective.constraints import G4CaloCostConstraint, G4CaloLengthConstraint
from examples.calo_multiobjective.simulation import G4CaloSimulation, G4CaloSimToReco

class UniformSampler(Sampler):
        def ask(self) -> torch.Tensor:
            values = {}
            for name, p in self._space:
                match p:
                    case Continuous():
                        u = np.random.random()
                        if p.log:
                            values[name] = float(p.lower * (p.upper / p.lower) ** u)
                        else:
                            values[name] = float(p.lower + u * (p.upper - p.lower))
                    case Integer():
                        values[name] = int(np.random.randint(p.lower, p.upper + 1))
                    case Categorical():
                        values[name] = np.random.choice(p.choices)
            return self._space.encode(values)


def _g4_calo_key(parts):
        return f"{parts[-1]}_{parts[-2]}_{parts[0].split('_')[-1]}"

calorimeter = ParameterSpace(
    key_fn=_g4_calo_key,
    **ParameterSpace.repeat('layer', 5,
        absorber=Group(
            thickness=Continuous(0.1, 50.0, log=True),
            material=Constant('G4_Fe'),
        ),
        scintillator=Group(
            thickness=Continuous(0.1, 50.0, log=True),
            material=Constant('G4_PbWO4'),
        ),
    )
)

if __name__ == "__main__":
    seed = 999
    constraint = MultiConstraint(G4CaloCostConstraint(max_cost=100_000),
                                 G4CaloLengthConstraint(max_length=200))
    
    SINGLE_CORE_CONFIG = SlurmWorkerConfig(processes=5, job_cpu=5, walltime="48:00:00", memory="48GB", partition="epyc-256")
    DEFAULT_GPU_CONFIG = SlurmWorkerConfig(job_cpu=4, gpu_resources="gpu:1", memory="16GB", walltime="48:00:00")

    simulation       = G4CaloSimulation("/home/nyl93mev/minicalosim_90baa21.sif").to(SINGLE_CORE_CONFIG)
    simulation_merge = MiniframeMergeTransform("output_df_*").to(SINGLE_CORE_CONFIG)
    sim2reco         = G4CaloSimToReco().to(SINGLE_CORE_CONFIG)
    reconstruction   = G4CaloReconstruction("/home/nyl93mev/minicalosim_90baa21.sif").to(DEFAULT_GPU_CONFIG)

    workflow = Workflow(simulation * 30 >> simulation_merge >> sim2reco >> reconstruction)
    workflow = optimize(workflow)
    workflow.visualize("workflow.png")
    
    objectives = [Objective(name="energy_resolution", minimize=True)] #Objective(name="particle_id", minimize=True)
    
    with AdaptiveHeterogeneousDaskSlurmWorker.from_workflow(workflow, logdir="logdir") as worker:
        candidate = calorimeter.sample(UniformSampler(), requires_grad=True)
        AidoBackend(f"calo_constrained", workflow, calorimeter, constraint, objectives, seed=seed).optimize(100, 20)


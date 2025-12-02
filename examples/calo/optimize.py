from argparse import ArgumentParser

from examples.calo.reconstruction import G4CaloReconstruction
from examples.calo.simulation import G4CaloSimulation, G4CaloSimToReco
from examples.calo.constraints import G4CaloCostConstraint, G4CaloLengthConstraint

from dsbo.workflow import Workflow
from dsbo.constraints import MultiConstraint
from dsbo.backends.optuna import OptunaBackend
from dsbo.simulation import DaskSimulationWrapper   
from dsbo.transforms import DataFrameMergeTransform
from dsbo.parameters import ContinuousDesignParameter, CategoricalDesignParameter, CompositeDesignParameter


class CaloDesignParameterization(CompositeDesignParameter):
    def __init__(self, name: str) -> None:
        parameters = []
        for i in range(3):
            absorber = CompositeDesignParameter(
                name=f"absorber_{i}",
                parameters=[ContinuousDesignParameter(f"thickness_absorber_{i}", 0.0, 100.0),
                            CategoricalDesignParameter(f"material_absorber_{i}", ["G4_Pb", "G4_Fe"])]
            )
            scintilator = CompositeDesignParameter(
                name=f"scintilator_{i}",
                parameters=[ContinuousDesignParameter(f"thickness_scintillator_{i}", 0.0, 100.0),
                            CategoricalDesignParameter(f"material_scintillator_{i}", ["G4_PbWO4", "G4_POLYSTYRENE"])]
            )
            parameters += [absorber, scintilator]

        super().__init__(name, parameters)

calo_design = CaloDesignParameterization("calo")


if __name__ == "__main__":
    seed = 1
    #argparse = ArgumentParser()

    transform = G4CaloSimToReco()
    base_simulation = G4CaloSimulation(workdir=f"files/run{seed}", container_path="/scratch/nyl93mev/aido/minicalosim_7829fde_2.sif")
    merge_transform = DataFrameMergeTransform("output_df_*")
    simulation = DaskSimulationWrapper(base_simulation=base_simulation, n_simulations=10, 
                                       n_workers=10, threads_per_worker=1,
                                       merge_function=merge_transform)
    reconstruction = G4CaloReconstruction(workdir=f"files/run{seed}", container_path="/scratch/nyl93mev/aido/minicalosim_7829fde_2.sif")
    constraint = MultiConstraint(G4CaloCostConstraint(max_cost=200_000),
                                 G4CaloLengthConstraint(max_length=200))

    workflow = Workflow(simulation=simulation,
                        postprocessing=transform,
                        reconstruction=reconstruction)

    OptunaBackend(f"calo_constrained_{seed}", workflow, calo_design, constraint, seed=seed).optimize(n_trials=500)
    simulation.close()
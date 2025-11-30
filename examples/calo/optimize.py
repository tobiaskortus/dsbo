from typing import Sequence
from dsbo.parameters import ContinuousDesignParameter, CategoricalDesignParameter, CompositeDesignParameter, DesignParameter


class CaloDesignParameterization(CompositeDesignParameter):
    def __init__(self, name: str) -> None:
        parameters = []
        for i in range(3):
            absorber = CompositeDesignParameter(
                name=f"absorber_{i}",
                parameters=[ContinuousDesignParameter(f"thickness_absorber_{i}", 1.0, 10.0),
                            CategoricalDesignParameter(f"material_absorber_{i}", ["G4_PbWO4", "G4_POLYSTYRENE"])]
            )
            scintilator = CompositeDesignParameter(
                name=f"scintilator_{i}",
                parameters=[ContinuousDesignParameter(f"thickness_scintillator_{i}", 1.0, 10.0),
                            CategoricalDesignParameter(f"material_scintillator_{i}", ["G4_Pb", "G4_Fe"])]
            )
            parameters += [absorber, scintilator]

        super().__init__(name, parameters)



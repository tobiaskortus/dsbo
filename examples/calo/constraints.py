from dsbo.constraints import GeometryConstraintBase


class G4CaloCostConstraint(GeometryConstraintBase):
    def __init__(self, max_cost: float) -> None:
        super().__init__(binary=False)
        self.max_cost = max_cost
        self.material_costs = {"G4_Pb": 25, "G4_Fe": 4.166,              #Absorber materials
                               "G4_PbWO4": 2500.0, "G4_POLYSTYRENE": 0.01}  #Scintillator materials

    def evaluate(self, design: dict): #-> Tuple[float]:
        total_cost = 0.0
        for parameter, value in design.items():
            if "material" in parameter:
                thickness_param = parameter.replace("material", "thickness")
                thickness = design[thickness_param]
                total_cost += self.material_costs[value] * thickness

        #return (max(0.0, total_cost - self.max_cost),)  
        return (total_cost - self.max_cost,)


class G4CaloLengthConstraint(GeometryConstraintBase):
    def __init__(self, max_length: float) -> None:
        super().__init__(binary=False)
        self.max_length = max_length

    def evaluate(self, design: dict) -> tuple[float]:  #-> Tuple[float]:
        total_length = 0.0
        for parameter, value in design.items():
            if "thickness" in parameter:
                total_length += value

        #return (max(0.0, total_length - self.max_length),)
        return (total_length - self.max_length,)
from logging import config
import os
import subprocess
import pandas as pd
from random import choice, uniform
from typing import Any, List, Sequence, Union, Optional
from xml.etree.ElementTree import Element, SubElement, tostring, ElementTree, indent

import tqdm

from dsbo.workflow import Workflow
from dsbo.constraints import GeometryConstraintBase, MultiConstraint
from dsbo.parameters import CategoricalDesignParameter, CompositeDesignParameter, ContinuousDesignParameter, DesignParameter, ValueRegistry
from dsbo.backends.utils.grpc.server import GrpcServer 

def get_experiment_of_design_callback():
    def callback(param: DesignParameter) -> Any:
        match param:
            case ContinuousDesignParameter():
                return uniform(param.min_value, param.max_value) 
            case CategoricalDesignParameter():
                return choice(param.choices)
            case _:
                if not issubclass(type(param), CompositeDesignParameter):
                    raise ValueError(f"Unsupported parameter type: {type(param)}")
    
    return callback


def split_constraints(constraints, return_dfs=True) -> List[pd.DataFrame]:
    return [pd.DataFrame(c) if return_dfs else c for c in zip(*constraints)]

class RoDeOBackend:
    def __init__(self, study_name: str, workflow: Workflow, 
                 design: Union[DesignParameter, CompositeDesignParameter],
                 constraint: Optional[GeometryConstraintBase],
                 seed: Optional[int] = 42) -> None:
        
        self.seed = seed
        self.design = design
        self.workflow = workflow
        self.constraint = constraint
        self.study_name = study_name
        self.dimension = 1
        self.study_type = "Optimization"
        self.bin_rodeo = "externals/RoDeO/build/rodeo"
        self.bin_grpc_client = "dsbo/backends/utils/grpc/client.py"

        self.train_file = "training_data.csv"

        self.workdir = workflow.simulation.workdir
        self.rodeo_workdir = os.path.join(self.workdir, "rodeo")

    def optimize(self, n_trials: int = 10):
        os.makedirs(self.rodeo_workdir, exist_ok=True)
        self.__generate_config_xml(n_trials)
        self.__generate_training_data()
        self.__run_rodeo()

    def __generate_training_data(self):
        sampler_callback = get_experiment_of_design_callback()

        designs = []
        objectives = []
        constraints = []

        for _ in tqdm.tqdm(list(range(10))):
            design_registry = ValueRegistry.sample_from(self.design, sampler_callback)
            design_vector = design_registry.to_design_vector()
            objective = self.workflow.run(design_registry.to_dict(True))
            
            if self.constraint is not None:
                c = self.constraint.evaluate(design_registry.to_dict(True))
                constraints.append(c)

            designs.append(design_vector)
            objectives.append(objective)

        designs = pd.DataFrame(designs)
        objectives = pd.DataFrame(objectives)
        training_data =pd.concat([designs, objectives], axis=1)
        training_data.to_csv(os.path.join(self.rodeo_workdir, self.train_file), index=False, header=False)
        #constraints = split_constraints(constraints, return_dfs=True)

        #designs.to_csv(os.path.join(self.rodeo_workdir, "designs.csv"), index=False, header=False)
        #objectives.to_csv(os.path.join(self.rodeo_workdir, "objectives.csv"), index=False, header=False)
        #if constraints is not None:
        #    for i, constraint in enumerate(constraints):
        #        constraint.to_csv(os.path.join(self.rodeo_workdir, f"constraints_{i}.csv"), index=False, header=False) 

    def __run_rodeo(self):
        def register_objective_callback(logger: Any):
            def objective_callback(design_vector: Sequence[float]) -> Sequence[float]:
                print("Evaluating objective for design vector:", design_vector)
                design_registry = ValueRegistry.from_design_vector(self.design, design_vector)
                objective_values = self.workflow.run(design_registry.to_dict(flatten=True))
                print(objective_values)
                return objective_values

            return objective_callback

        def register_constraint_callback(logger: Any):
            def constraint_callback(design_vector: Sequence[float]) -> Sequence[float]:
                print("Evaluating constraints for design vector:", design_vector)
                design_registry = ValueRegistry.from_design_vector(self.design, design_vector)
                constraint_values = self.constraint.evaluate(design_registry.to_dict(flatten=True))
                return constraint_values

            return constraint_callback if self.constraint is not None else None

        objective_callback = register_objective_callback(None)
        constraint_callback = register_constraint_callback(None)

        with GrpcServer(objective_callback, constraint_callback) as server:
            config_file = os.path.join(self.rodeo_workdir, "config.xml")
            rodeo_process = subprocess.Popen([self.bin_rodeo, config_file], cwd=os.getcwd())
            rodeo_process.wait()
    
    def __generate_config_xml(self, n_trials: int):

        def add_parameter_element(root: Element, name: str, lower: float, upper: float):
            parameter = SubElement(root, "parameter")
            SubElement(parameter, "name").text = name
            SubElement(parameter, "type").text = "REAL"
            SubElement(parameter, "lower_bound").text = str(lower)
            SubElement(parameter, "upper_bound").text = str(upper)

        #def add_constraint_element(root: Element, constraint: GeometryConstraintBase):
        #    constraint_fn = SubElement(root, "constraint_function")
        #    SubElement(constraint_fn, "name").text = str(constraint)
        #    SubElement(constraint_fn, "constraint_type").text = "GEOMETRY"
        #    SubElement(constraint_fn, "constraint_value").text = "0.0"
        #    SubElement(constraint_fn, "executable_filename").text = self.bin_grpc_client
        #    SubElement(constraint_fn, "output_filename").text = "TEMPORARY FILE"
        #    SubElement(constraint_fn, "training_data_filename").text = "TEMPORARY FILE"
        #    SubElement(constraint_fn, "design_vector_filename").text = "TEMPORARY FILE"

        root = Element("optimization_study")
        general_settings = SubElement(root, "general_settings")

        SubElement(general_settings, "name").text = self.study_name
        SubElement(general_settings, "type").text = self.study_type
        SubElement(general_settings, "dimension").text = str(self.design.num_parameters()) #REPLACE WITH NUMBER OF PARAMETERS
        SubElement(general_settings, "number_of_function_evaluations").text = str(n_trials)
        SubElement(general_settings, "max_number_of_inner_iterations").text = "200000"
        SubElement(general_settings, "number_of_threads").text = "4"
        SubElement(general_settings, "display").text = "ON"

        optimization_parameters = SubElement(root, "optimization_parameters")
        for param in self.design.iterate_flatten():
            if isinstance(param, ContinuousDesignParameter):
                add_parameter_element(optimization_parameters, param.name, param.min_value, param.max_value)
            if isinstance(param, CategoricalDesignParameter):
                for i, choice in enumerate(param.choices):
                    add_parameter_element(optimization_parameters, f"{param.name}_{choice}", 0, 1)

        objective_fn = SubElement(root, "objective_function")
        SubElement(objective_fn, "name").text = self.study_name
        SubElement(objective_fn, "executable_filename").text = f"{self.bin_grpc_client}" # -w {self.rodeo_workdir}
        SubElement(objective_fn, "output_filename").text = os.path.join(os.getcwd(), self.rodeo_workdir, "objective.dat")
        SubElement(objective_fn, "training_data_filename").text = os.path.join(os.getcwd(), self.rodeo_workdir, self.train_file)
        SubElement(objective_fn, "design_vector_filename").text = os.path.join(os.getcwd(), self.rodeo_workdir, "dv.dat")
        SubElement(objective_fn, "number_of_training_iterations").text = "10000"

        #if self.constraint is not None:
        #    if isinstance(self.constraint, MultiConstraint):
        #        for constr in self.constraint.constraints:
        #            add_constraint_element(root, constr)
        #    else:
        #        add_constraint_element(root, self.constraint)

        tree = ElementTree(root)
        indent(tree, space="  ", level=0)

        with open(os.path.join(self.rodeo_workdir, "config.xml"), "wb") as f:
            tree.write(f, encoding="utf-8", xml_declaration=False)
            

import os
import sys
import time
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


def get_absolute_dir(workdir: str) -> str:
    if os.path.isabs(workdir):
        return workdir
    return os.path.join(os.getcwd(), workdir)

def get_filepath(workdir, filename: str) -> str:
    return os.path.join(get_absolute_dir(workdir), filename)

class RoDeOBackend:
    def __init__(self, study_name: str, workflow: Workflow, 
                 design: Union[DesignParameter, CompositeDesignParameter],
                 constraint: Optional[Union[GeometryConstraintBase, MultiConstraint]] = None,
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

        self.workdir = workflow.simulation.workdir
        self.rodeo_workdir = os.path.join(self.workdir, "rodeo")

    def optimize(self, n_trials: int = 10):
        os.makedirs(self.rodeo_workdir, exist_ok=True)
        self.__generate_config_xml(n_trials)
        self.__generate_training_data()
        self.__run_rodeo()

    def __generate_training_data(self):
        """Generate training data (design of experiment) for the RoDeO optimization process.
        All generated data (objectives and constraints) is saved in the RoDeO working directory (workdir/rodeo).
        """
        sampler_callback = get_experiment_of_design_callback()

        designs = []
        objectives = []
        constraints = []

        for _ in tqdm.tqdm(list(range(30))):
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
        objective_training_data = pd.concat([designs, objectives], axis=1)
        objective_training_data.to_csv(os.path.join(self.rodeo_workdir, "training_data.csv"), index=False, header=False)
        
        constraints = split_constraints(constraints, return_dfs=True)
        if constraints is not None:
            for i, constraint in enumerate(constraints):
                if isinstance(self.constraint, MultiConstraint):
                    constraint_name = self.constraint.constraints[i]

                constraint_training_data = pd.concat([designs, constraint], axis=1)
                constraint_training_data.to_csv(os.path.join(self.rodeo_workdir, f"constraints_{constraint_name}.csv"), index=False, header=False) 


    def __run_rodeo(self):        
        def register_objective_callback(logger: Any):
            def objective_callback(design_vector: Sequence[float]) -> Sequence[float]:
                start_time = time.time()
                design_registry = ValueRegistry.from_design_vector(self.design, design_vector)
                objective_values = self.workflow.run(design_registry.to_dict(flatten=True))
                elapsed_time = time.time() - start_time
                print(f"Objective evaluation took {elapsed_time:.4f} seconds and returnded {float(objective_values[0])}")
                return objective_values

            return objective_callback

        def register_constraint_callback(logger: Any):
            def constraint_callback(design_vector: Sequence[float], name: str) -> Sequence[float]:
                start_time = time.time()
                design_registry = ValueRegistry.from_design_vector(self.design, design_vector)

                if isinstance(self.constraint, MultiConstraint):
                    constraints = self.constraint.get_by_name(name).evaluate(design_registry.to_dict(flatten=True))
                else:
                    constraints = self.constraint.evaluate(design_registry.to_dict(flatten=True))
                
                elapsed_time = time.time() - start_time
                print(f"Constraint evaluation took {elapsed_time:.4f} seconds and returnded {float(constraints[0])}")
                return constraints

            return constraint_callback if self.constraint is not None else None

        objective_callback = register_objective_callback(None)
        constraint_callback = register_constraint_callback(None)

        with GrpcServer(objective_callback, constraint_callback) as server:
            config_file = os.path.join(self.rodeo_workdir, "config.xml")
            rodeo_process = subprocess.Popen([self.bin_rodeo, config_file], cwd=os.getcwd())
            rodeo_process.wait()
    
    def __generate_config_xml(self, n_trials: int):
        """ Generate a custom RoDeO configuration XML file to match the specified parameters.
        The resulting XML file is saved in the RoDeO working directory (workdir/rodeo).

        Args:
            n_trials (int): Number of trials (blackbox function evaluations) for the optimization study.
        """
        def add_parameter_element(root: Element, name: str, lower: float, upper: float):
            parameter = SubElement(root, "parameter")
            SubElement(parameter, "name").text = name
            SubElement(parameter, "type").text = "REAL"
            SubElement(parameter, "lower_bound").text = str(lower)
            SubElement(parameter, "upper_bound").text = str(upper)

        def add_ojective_element(root: Element, objective: Any):
            pass

        def add_constraint_element(root: Element, constraint: GeometryConstraintBase):
            constraint_fn = SubElement(root, "constraint_function")
            SubElement(constraint_fn, "name").text = str(constraint)
            SubElement(constraint_fn, "constraint_type").text = "lt"
            SubElement(constraint_fn, "constraint_value").text = "0.0"
            SubElement(constraint_fn, "executable_filename").text = get_constraint_executable(str(constraint))
            SubElement(constraint_fn, "output_filename").text = get_filepath(self.rodeo_workdir, f"constraints_{str(constraint)}.dat")
            SubElement(constraint_fn, "training_data_filename").text = get_filepath(self.rodeo_workdir, f"constraints_{str(constraint)}.csv") #DIFFERENT TRAINING FILES!!!???
            SubElement(constraint_fn, "design_vector_filename").text = get_filepath(self.rodeo_workdir, "dv.dat")

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

        def get_constraint_executable(constraint_name: str) -> str:
            """ Construct the command to execute the gRPC client for constraint evaluation."""
            executable =  f"{sys.executable} {get_absolute_dir(self.bin_grpc_client)}"
            executable += f" -w {get_absolute_dir(self.rodeo_workdir)}"
            executable += f" -a localhost:50051"
            executable += f" -q {constraint_name}" #query
            return executable

        def get_objective_executable() -> str:
            """ Construct the command to execute the gRPC client for objective evaluation."""
            executable =  f"{sys.executable} {get_absolute_dir(self.bin_grpc_client)}"
            executable += f" -w {get_absolute_dir(self.rodeo_workdir)}"
            executable += f" -a localhost:50051"
            executable += f" -q objective" #query
            return executable

        objective_fn = SubElement(root, "objective_function")
        SubElement(objective_fn, "name").text = self.study_name
        SubElement(objective_fn, "executable_filename").text = get_objective_executable()
        SubElement(objective_fn, "output_filename").text = get_filepath(self.rodeo_workdir, "objective.dat")
        SubElement(objective_fn, "training_data_filename").text = get_filepath(self.rodeo_workdir, "training_data.csv")
        SubElement(objective_fn, "design_vector_filename").text = get_filepath(self.rodeo_workdir, "dv.dat")
        SubElement(objective_fn, "number_of_training_iterations").text = "10000"

        if self.constraint is not None:
            if isinstance(self.constraint, MultiConstraint):
                for constr in self.constraint.constraints:
                    add_constraint_element(root, constr)
            else:
                add_constraint_element(root, self.constraint)

        tree = ElementTree(root)
        indent(tree, space="  ", level=0)

        with open(os.path.join(self.rodeo_workdir, "config.xml"), "wb") as f:
            tree.write(f, encoding="utf-8", xml_declaration=False)
            

# Project Generator CLI

DSBO includes an interactive command-line tool for creating module packages and
multi-module workflow projects. The generator creates the package structure,
starter implementations, tests, and `pyproject.toml` needed to begin development.

## Motivation

The CLI standardizes how DSBO projects are organized. Generated projects follow
the same package layout, naming conventions, test structure, and base-class
interfaces, making modules and workflows compatible with DSBO tooling and easier
to combine without project-specific integration work.

This consistency also creates a "learn one, navigate all" effect: once someone
understands one generated DSBO project, they can quickly find the modules,
workflow definition, configuration, and tests in any other generated project.
The generated structure is therefore both a starting point and a shared contract
between project authors, users, and DSBO.

## Installation

The `dsbo-cli` command is installed with DSBO. For local development, install the
repository in editable mode:

```sh
pip install -e .
```

## Creating a Project

Run the generator from the directory where the new project should be created:

```sh
dsbo-cli
```

The CLI prompts for a project type and a Python package name, then displays the
planned files before writing anything. Project and module names must be valid
Python identifiers. The generator will not overwrite an existing directory.

### Module Project

Choose **Module** to create a package containing one reusable processing unit.
The CLI asks for:

- The project and package name, such as `detector_simulation`
- The class name, such as `DetectorSimulation`
- The DSBO base type to implement, such as `Module`, `SimulationBase`, or
  `ReconstructionBase`

The generated project has this structure:

```text
detector_simulation/
├── pyproject.toml
├── detector_simulation/
│   ├── __init__.py
│   └── detector_simulation.py
└── tests/
    └── autotests_module.py
```

### Workflow Project

Choose **Workflow** to create a project composed of multiple modules. Add each
module by specifying its class name, file name, and base type. The setup menu
also lets you edit or remove entries before finishing.

Several classes can share a Python file by assigning them the same file name.
The review screen highlights shared files before generation.

For example, a workflow with `Simulation` in `simulation.py` and `Reconstruction`
in `reconstruction.py` produces:

```text
detector_workflow/
├── pyproject.toml
├── detector_workflow/
│   ├── __init__.py
│   ├── workflow.py
│   └── modules/
│       ├── __init__.py
│       ├── simulation.py
│       └── reconstruction.py
└── tests/
    ├── autotests_workflow.py
    ├── autotests_simulation.py
    └── autotests_reconstruction.py
```

## Reviewing and Installing

Before creating the project, the CLI shows its destination, configured classes,
base types, and generated file tree. Confirm the summary to generate the files,
or cancel with **Ctrl+C** at any prompt without creating the project.

After generation, enter the project directory and install it in editable mode:

```sh
cd detector_workflow
pip install -e .
```

The generated implementations contain guidance and placeholders for the methods
required by the selected base types. Replace those placeholders with the project
logic, then run the generated tests while developing.

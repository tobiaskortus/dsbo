# DSBO — Distributed Surrogate-Based Optimization

DSBO is a Python framework for **composable, distributed simulation workflows** for particle detector design optimization. Express a detector simulation and reconstruction pipeline as a graph of modular components, then execute that graph on a Dask cluster with automatic dependency scheduling, parallel execution, and per-module resource assignment.

## Why DSBO

Detector design requires evaluating thousands of candidate geometries through expensive Monte Carlo simulations. DSBO makes this practical by modeling the optimization as a configurable, distributed pipeline:

- **Declarative workflow definition**: The Python DSL (`>>`, `|`, `*`) lets you define complex simulation-to-reconstruction pipelines in a few lines of code. The framework handles execution order, parallelization, and data dependencies automatically.

- **Scales from laptop to cluster**: Run the same workflow without a worker for local debugging, then parallelize across a Dask cluster without any code changes. Per-module CPU/GPU annotations let you route, e.g., simulations to CPU and reconstruction to GPU nodes.

- **Unified surrogate plug-in**: A unified `patch()` interface assigns surrogates to either individual modules or entire workflows. The surrogate handles trust-regions and automatic re-training, while optimization algorithms interact with the workflow exactly as if no surrogate was present.

## Installation

DSBO is not yet available on PyPI. Install directly from source or GitHub:

#### From source

Clone the repository and install in editable mode (recommended for development):

```sh
git clone git@github.com:tobiaskortus/dsbo.git
cd dsbo
pip install -e .
```

#### From GitHub

Install directly from the repository without cloning:

```sh
pip install git+https://github.com/tobiaskortus/dsbo
```

**Requirements:** Python 3.10+, PyTorch 2.0+. For GPU support, install PyTorch with CUDA before installing DSBO.

## Quick Start

Define simulation and reconstruction modules, wire them into a workflow, and evaluate design candidates:

```python
from dsbo.workflow import Workflow

# Compose modules into a workflow graph
wf = Workflow(sim >> reco >> analysis)

# Run for a design candidate
candidate = space.sample()
result = wf.run(candidate)
# result: {"energy_resolution": 0.12, "cost": 4500.0}
```

Workflows can be executed locally for debugging or distributed across a heterogeneous Dask cluster without any code changes — simply wrap the call in a worker context.

!!! note "Getting started"

    - [Writing Modules](guide/modules.md) covers all module types including simulation, reconstruction, and differentiable modules
    - [Building Workflows](guide/workflows.md) explains replication, fan-in, and parallel execution patterns
    - [I/O and Autograd](guide/io.md) details the workdir system and gradient tracking through `NumpyFile`

## Documentation Structure

| Section | Content |
|---------|---------|
| **Core Concepts** | How the system works at a conceptual level: modules, workflows, surrogates |
| **Guide** | Step-by-step instructions for writing modules, managing I/O, building workflows, and using surrogates |
| **API Reference** | Precise signatures, method tables, and implementation details |
| **Examples** | Real detector optimization pipelines (G4Calo, HepEmShow, Proton CT) |
| **Troubleshooting** | Solutions for common errors and issues |

## Project Status

!!! warning

    dsbo is prototype software! If you notice any issues or have any suggestions for improvements, please write up a GitHub issue detailing the problem or proposal. We also recommend “watching” the GitHub repository to keep track of new releases, and upgrading promptly when they occur.

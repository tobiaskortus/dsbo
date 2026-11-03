# Parameter Spaces and Candidates

`ParameterSpace` defines the inputs of an optimization study, including which quantities vary and their sampling ranges. A `Candidate` represents one configuration to evaluate. Together, they provide a common input interface for reference configurations, sampled candidate sets, and configurations proposed by an optimizer.

## Defining the Design Space

A parameter space associates each input name with a domain:

```python
from dsbo.parameters import Continuous, ParameterSpace

space = ParameterSpace(
    x=Continuous(0.0, 1.0),
    y=Continuous(-1.0, 1.0),
)
```

| Domain | Values |
| --- | --- |
| `Continuous(lower, upper)` | Real-valued scalar |
| `Integer(lower, upper)` | Integer sampled from the inclusive range |
| `Categorical(choices)` | One of the supplied choices |
| `Vector(lower, upper, size)` | Fixed-length continuous vector with shared component bounds |
| `Constant(value)` | Fixed input, excluded from optimization |

!!! note "Bounds and Feasibility"

    The bounds determine how values are sampled; they do not constrain explicit assignments or optimizer updates. Bound enforcement and coupled feasibility constraints must be handled by the optimization code.

## Structured Inputs

### Vectors

`Vector(lower, upper, size)` declares an array-valued input with shared component bounds. `Candidate.from_values()` accepts a sequence under the vector's name:

```python
from dsbo.parameters import Candidate, Vector

vector_space = ParameterSpace(x=Vector(0.0, 1.0, size=3))
vector_candidate = Candidate.from_values(
    vector_space,
    {"x": [0.2, 0.4, 0.6]},
)

vector_candidate.values["x"]  # tensor([0.2000, 0.4000, 0.6000])
```

Individual component names, `x_1` through `x_3`, are also accepted by `from_values()`. The decoded dictionary regroups these entries under `x`.

### Groups

`Group` namespaces related declarations. Group paths become dot-separated keys in both `from_values()` and `values`, rather than nested dictionaries:

```python
from dsbo.parameters import Group

grouped_space = ParameterSpace(
    a=Group(
        x=Continuous(0.0, 1.0),
        y=Continuous(-1.0, 1.0),
    ),
)
grouped_candidate = Candidate.from_values(
    grouped_space,
    {"a.x": 0.5, "a.y": 0.0},
)
```

Groups can be nested. The naming convention is controlled by `ParameterSpace`'s `key_fn`, which receives a tuple of path components. For example, `key_fn="/".join` produces `a/x` instead of `a.x`. Custom formatters must preserve distinct parameter names.

### Repeated Groups

`ParameterSpace.repeat()` generates numbered groups from a shared declaration. Repeated inputs have independent values; the declaration does not tie them together:

```python
repeated_space = ParameterSpace(
    **ParameterSpace.repeat(
        "group",
        3,
        x=Continuous(0.0, 1.0),
        y=Continuous(-1.0, 1.0),
    ),
)
```

The resulting names range from `group_0.x` and `group_0.y` to `group_2.x` and `group_2.y`. Explicit candidates use these names, as with any other grouped input.

## A Mixed Space

A space can combine continuous, integer, categorical, vector, and constant inputs:

```python
import torch

from dsbo.parameters import (
    Candidate,
    Categorical,
    Constant,
    Continuous,
    Integer,
    ParameterSpace,
    Vector,
)

space = ParameterSpace(
    beam_energy=Continuous(10.0, 100.0),
    n_layers=Integer(2, 12),
    absorber=Categorical(["Pb", "Fe", "Cu"]),
    cell_offsets=Vector(0.0, 1.0, 4),
    run_tag=Constant("run-2026"),
)
```

## Sampling Designs

Sampling provides an initial search population, a parameter scan, or a training set for a surrogate. A sampler can be registered on the space for repeated requests or supplied directly to an individual sampling call. `sample()` returns one candidate; `sample_batch()` requests a set of candidates from the same sampler:

```python
from dsbo.parameters import SobolSampler

space.register_sampler(SobolSampler(seed=42))
candidate = space.sample()
initial_designs = space.sample_batch(batch_size=16)
```

Reusing the registered instance advances its sequence. Recreating a sampler with the same seed instead repeats the sequence, which matters when extending an existing sample. dsbo provides the following sampling strategies, all imported from `dsbo.parameters`:

| Sampler | Sampling Strategy |
| --- | --- |
| `UniformSampler(seed=None)` | Independent uniform draws |
| `SobolSampler(seed=None, scramble=True)` | Successive points from a low-discrepancy Sobol sequence |
| `LatinHypercubeSampler(seed=None)` | One point per stratum in each dimension, within a sampling call |

A separate Latin hypercube design can be generated with a sampler supplied for that batch:

```python
from dsbo.parameters import LatinHypercubeSampler

lhs_designs = space.sample_batch(
    sampler=LatinHypercubeSampler(seed=7),
    batch_size=16,
)
```

Latin hypercube stratification applies within a single sampling call, not across repeated calls to `sample()`. The complete design is therefore generated with `sample_batch()`. The explicitly supplied sampler does not replace the registered Sobol sampler.

The returned `CandidateBatch` can be passed to [`workflow.run_batch()`](workflows.md#batch-evaluation).

## Explicit Candidates

`Candidate.from_values()` constructs a reference configuration or a known starting point. Every variable input is required by name; fixed inputs are taken from the space:

```python
from dsbo.parameters import Candidate

reference = Candidate.from_values(
    space,
    {"x": 0.5, "y": 0.0},
)
```

An existing workflow evaluates this configuration with `workflow.run(reference)` and supplies `reference.values` to its modules as the `design` dictionary. The module interface is described in [Writing Modules](modules.md), and evaluation in [Building Workflows](workflows.md).

Existing candidates can also be combined, for example to include the reference configuration in the same evaluation set as the sampled designs:

```python
from dsbo.parameters import CandidateBatch

evaluation_set = CandidateBatch([reference, *initial_designs])
```

## Connecting an Optimizer

For continuous optimization with PyTorch, the optimizer operates on the candidate tensor, with gradient tracking enabled at candidate construction:

```python
import torch

current_design = Candidate.from_values(
    space,
    {"x": 0.5, "y": 0.0},
    requires_grad=True,
)
optimizer = torch.optim.Adam([current_design.tensor], lr=0.01)
```

The workflow evaluates `current_design`, and the optimizer updates its tensor between evaluations. Modules continue to receive the current named values. The workflow must provide derivatives for this use; [Gradient-Based Optimization](gradient-based-optimization.md) gives a complete loop, including the module backward implementations required for a file-based computation.

Mixed spaces can be sampled and evaluated through the same workflow interface, but their integer and categorical inputs are not made differentiable by enabling gradient tracking. A search algorithm that proposes discrete configurations can construct them with `Candidate.from_values()` instead.

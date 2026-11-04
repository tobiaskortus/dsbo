# Stochastic Execution and Reproducibility

`dsbo` separates two sources of randomness: the process-wide generators used by Python, NumPy, and PyTorch, and the workflow's own module-seed allocator. The utilities in `dsbo.utils.random` control both.

## Seeding Global Generators

`seed_all()` seeds Python's `random`, NumPy, PyTorch on CPU and CUDA when available, and the workflow random engine:

```python
from dsbo.utils.random import seed_all

seed_all(2026)
```

Calling `seed_all()` again with the same value restarts every seeded generator from that value. It does not configure deterministic algorithm selection, thread counts, or the random-number generators of external simulators or containers. Those must be configured separately, for example with `torch.use_deterministic_algorithms(True)`, `torch.backends.cudnn.deterministic`, `torch.set_num_threads(1)`, or a fixed simulator seed passed into the module.

## Seeding Module Assignment Only

To control workflow module-seed allocation without touching the global generators, use `seed_random_engine()`:

```python
from dsbo.utils.random import seed_random_engine

seed_random_engine(2026)
```

`seed_random_engine()` returns the current `ModuleRandomEngine` singleton. Passing the same seed returns the existing engine without reseeding it; passing `None` creates an engine with a fresh 64-bit seed drawn from `SystemRandom`.

`get_random_engine()` returns the active engine. The engine is initialized lazily, so it may not exist until the first execution or explicit seed call:

```python
from dsbo.utils.random import get_random_engine

engine = get_random_engine()
engine.seed_all(workflow)  # assign module seeds immediately
```

Without explicit initialization, the engine draws its initial seed from `SystemRandom`. Setting only Python's global `random` seed does not initialize it, because `SystemRandom` is independent of the global generator. `seed_all()` initializes it.

| Utility | Seeds |
| --- | --- |
| `seed_all(seed)` | Python `random`, NumPy, PyTorch CPU/CUDA, workflow engine |
| `seed_random_engine(seed)` | Workflow engine only |
| `get_random_engine()` | Returns the workflow engine; does not seed it |

Samplers and surrogate strategies with their own generators use their own seed arguments, as described in [Sampling Designs](parameters.md#sampling-designs) and [Surrogate Strategies](surrogates.md#surrogate-strategies). Their sequences are independent of the workflow engine.

## Module Seed Assignment

Before each forward execution, `run()` and `run_batch()` call `get_random_engine().seed_all(workflow)`. The engine resets its sequence to its stored seed, then walks `workflow.task_graph.nodes` and calls `set_seed()` on each node with a 64-bit value from that sequence.

The walk follows the order in which modules were inserted into the task graph, not topological or depth order. Replicas are inserted in replication order and are separate graph nodes, so each replica receives a distinct seed and is identified by its `.i` index. Consequently:

- The same engine seed and the same graph construction order produce the same module seeds on every execution.
- The engine sequence restarts on each execution; it does not advance across runs.
- Changing construction order, replication counts, or module insertion order changes which seed a task receives.

The workflow assigns `self._seed` before each execution, replacing any seed set earlier. Use this value to seed any random generators or external programs inside your module:

```python
import random

from dsbo.modules import Module


class Simulation(Module):
    def __call__(self, design):
        rng = random.Random(self._seed)
        native_seed = rng.randrange(2**31)
        # pass native_seed to the simulator
```

Each replica already receives its own seed, so incorporating `self.i` is not required.

Dynamic submodules have their own private attributes, while their forward methods execute on the parent. A submodule's `_seed` and its parent's `_seed` are therefore separate, and each graph node is seeded independently.

## Distributed Execution

Seeds are assigned on the driver before tasks are scheduled. Each task carries its module's `_seed` to the worker, so worker-local generator state does not affect the assigned value. Modules that derive native seeds from `_seed` therefore behave consistently across local and distributed execution.

## Reproducibility During Backpropagation

When a derivative requires the same random samples as the forward pass, the module's `backward()` implementation must reproduce them using the same seeds and simulation settings. The framework provides module seeds but does not automatically restore random generators or simulator state.

The backward pass reuses the module's stored seed. Complete `.backward()` before starting another forward execution of the same workflow to avoid overwriting state needed for replay.

See [Reverse and Forward Gradient Flow](gradient-flow.md) for derivative execution.

## Reproducibility Checklist

- Call `seed_all(seed)` once before a study, or `seed_random_engine(seed)` when only module assignment should be controlled.
- Keep workflow construction order, replication counts, and worker or chunk counts fixed across comparable runs.
- Derive native seeds inside modules from `self._seed`; do not rely on constructor seeds.
- Replay the same seeds, event order, and partitioning in `backward()`.
- Configure external simulators, containers, and deterministic PyTorch settings separately from `dsbo` seeding.
- Evaluate final designs on independent seed streams when optimizing a sample-average approximation.

# Module Templates and Dynamic Submodules

Module templates, encoded tensor access, and dynamic-submodule backward implementations extend the [stateless module](modules.md) interface. Dynamic submodules expose individual methods as separate workflow stages; the [workflow guide](workflows.md#dynamic-submodules) covers their composition and scheduling.

## Pre-Defined Module Structures

For common execution patterns, dsbo provides base classes that implement `__call__` and delegate to more specific methods. They are used when their lifecycle matches the computation; otherwise `Module` is subclassed directly.

### Simulation Module

`SimulationBase` (`dsbo.modules.simulation`) provides an interface for data-generating stages. Subclasses implement `simulate(design, **kwargs)`; the base class calls this method and returns an empty result dictionary. Generated data is shared with downstream modules through registered output files.

### Reconstruction Module (Train/Validate Pattern)

`ReconstructionBase` (`dsbo.modules.reconstruction`) provides an interface for stages that train and evaluate a reconstruction model. Subclasses implement `train(design, **kwargs)` and `validate(design, **kwargs)`. The base class calls them in that order on the same instance within one task, and the dictionary returned by `validate()` becomes the module result.

## Accessing Encoded Coordinates

Modules normally receive decoded design values. Setting `requires_design_tensor = True` makes the encoded coordinates available to a computation, such as evaluating a model defined in that representation:

```python
from dsbo.modules import Module

class EncodedDesign(Module):
    requires_design_tensor = True

    def __call__(self, design, design_tensor=None, **kwargs):
        return {"coordinates": design_tensor}
```

The runtime also supplies this argument to the module's backward method when requested. The flag provides tensor access but does not implement derivative propagation. The backward contract is described in [Differentiable Workflows](autograd.md).

## Module with Dynamic Submodules

The [workflow guide](workflows.md#dynamic-submodules) introduces composition and scheduling of dynamic submodules. For differentiation, each exposed method can register its own backward implementation with `@method.backward`:

```python
from dsbo.modules import Module, dynamic_submodule
from dsbo.workflow import Workflow

class MyModule(Module):
    @dynamic_submodule
    def step(self, design, **kwargs):
        # Perform this stage's computation and return a result dictionary.
        pass

    @step.backward
    def step_backward(self, design, grad_output):
        # Propagate the incoming gradients through this stage.
        pass


module = MyModule()
workflow = Workflow(module.step)
```

Each backward implementation is registered with `@method.backward` and has the signature `(self, design, grad_output)`. A distinct Python method name, such as `step_backward`, is used rather than replacing the parent's `backward` method. These methods are only needed for workflow differentiation; the reverse-pass contract and gradient-enabled candidate setup are described in [Autograd](autograd.md).

!!! note "Attribute delegation"

    Dynamic submodules delegate attributes and file registrations to their parent. All of the parent's registered inputs and outputs are exposed by each submodule.

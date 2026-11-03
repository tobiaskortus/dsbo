# Writing Surrogates

A **surrogate** is a trained model that approximates an expensive computation — either an entire workflow or a single module — so the optimizer can evaluate candidates cheaply. Surrogates are plugged in with a unified `patch()` interface: optimization algorithms interact with the workflow exactly as if no surrogate were present.

This guide builds on [Gradient-Based Optimization](gradient-based-optimization.md). The [surrogate-assisted walkthrough](surrogate-assisted-optimization.md) applies these interfaces to that same problem. Lifecycle tasks use [dynamic submodules](module-templates-and-dynamic-submodules.md#module-with-dynamic-submodules).

dsbo distinguishes two kinds of surrogates:

| Kind | Replaces | Trained on | Predicts |
|---|---|---|---|
| **Workflow surrogate** | The entire workflow | Recorded workflow results | Objective values from a design |
| **Module surrogate** | A single module | The module's own output files | The module's output files from its inputs |

## Fundamentals

Every surrogate has four components:

1. **A `SurrogateStrategy`** — the `strategy` argument controls when the surrogate is trusted and how training candidates are generated (see [Surrogate Strategies](#surrogate-strategies))
2. **Lifecycle tasks** — `train`, or `pretrain` + `finetune`, decorated with `@dynamic_submodule`
3. **`state_dict()` and `from_state_dict()`** — serialize and restore the trained model for artifact storage and on-demand loading
4. **`__call__`** — predicts the surrogate's outputs for a design

```python
from dsbo.surrogates import SurrogateBase, SurrogateStrategy
from dsbo.modules import dynamic_submodule
from dsbo.utils.io import Input, Workdir

class MySurrogate(SurrogateBase):
    def __init__(self, strategy: SurrogateStrategy):
        super().__init__(strategy=strategy)
        self.data = Input(Workdir.register_file("training", "data.npz"))

    @dynamic_submodule
    def train(self, design, candidates):
        model = fit_model(self.data(), candidates)
        return {}

    def state_dict(self) -> dict:
        return {"model": self.model.state_dict()}

    def from_state_dict(self, state: dict) -> None:
        self.model.load_state_dict(state["model"])

    def __call__(self, design, **kwargs):
        return {"energy_resolution": self.model(design)}
```

!!! note "Surrogates are stateless"

    Like modules, surrogates must not depend on task-time mutations being visible to the next task. Persistent model state goes through `state_dict()` / `from_state_dict()` and [artifact storage](artifacts.md#surrogate-checkpoints), so it can be reloaded independently of the process executing the task.

## Surrogate Strategies

A `SurrogateStrategy` decides *when* the surrogate may be used and *what data it trains on*. Subclasses implement `is_in_trust_region()` and `generate_candidates()`:

```python
import torch
from dsbo.parameters import Candidate, CandidateBatch
from dsbo.surrogates import SurrogateStrategy

class LocalStrategy(SurrogateStrategy):
    """Samples candidates locally around the last evaluated design."""

    def __init__(self, num_candidates: int = 20, radius: float = 1.0):
        self.num_candidates = num_candidates
        self.radius = radius
        self.center: torch.Tensor | None = None

    def is_in_trust_region(self, candidate: Candidate) -> bool:
        if self.center is None:
            return False
        return bool((candidate.tensor - self.center).norm() < self.radius)

    def generate_candidates(self, candidate, space, finetune=False) -> CandidateBatch:
        self.center = candidate.tensor.detach().clone()
        return CandidateBatch([
            Candidate(space, self.center + 0.1 * torch.randn_like(self.center))
            for _ in range(self.num_candidates)
        ])
```

- `is_in_trust_region` — decides whether the surrogate is trusted to predict for a candidate, e.g. by measuring the distance to the last evaluated design. It is only consulted once a trained model exists; before that, the surrogate is never used. The candidate's design tensor is available via `candidate.tensor`.
- `update` — optional hook called immediately before `generate_candidates` when a training session starts; it can update the strategy's internal state, such as the trust-region center or covariance.
- `generate_candidates` — called when (re-)training is due. It receives the candidate that triggered training and the parameter space, and typically draws the training batch around that candidate. The `finetune` flag is `True` when the finetune stage is the selected lifecycle task, allowing the strategy to focus the training batch near already-explored regions.

## Training Lifecycle

Training is fully automatic and identical for workflow and module surrogates. For a workflow with a patched surrogate, dsbo checks before every evaluation whether the surrogate needs (re-)training:

1. No trained model artifact exists, **or** the candidate is outside the trust region → a training session starts.
2. The strategy's `generate_candidates` produces a training batch, and the real workflow is evaluated on that batch to collect training data.
3. The selected lifecycle task runs, and the resulting model is published to artifact storage.
4. Subsequent candidates inside the trust region are evaluated with the surrogate; the trained model is reused from storage instead of re-trained.

When both `pretrain` and `finetune` are implemented, pretraining runs once and every later training session becomes a finetune. When only `train` is implemented, every training session retrains from scratch.

```python
from dsbo.autograd import train_op

class MySurrogate(SurrogateBase):
    @train_op
    @dynamic_submodule
    def pretrain(self, design, candidates):
        ...  # train from scratch

    @train_op
    @dynamic_submodule
    def finetune(self, design, candidates):
        ...  # continue from the pretrained model
```

!!! note "Training loops"

    Lifecycle tasks that contain a PyTorch training loop require `@train_op`, stacked above `@dynamic_submodule`.

## Workflow Surrogates

A workflow surrogate replaces the whole workflow: it maps a design directly to the objective values the workflow would return. Workflow surrogates extend `WorkflowSurrogate` (module surrogates extend `ModuleSurrogate`). The workflow's `patch()` method attaches the surrogate:

```python
from dsbo.workflow import Workflow

wf = Workflow(sim >> reco)
wf.patch(MyWorkflowSurrogate(MyStrategy()))

result = wf.run(candidate)
# result: {"energy_resolution": ..., "cost": ...}  — predicted, not simulated
```

The `__call__` method returns a `dict[str, Tensor]` keyed by the workflow's objective names. The `design_tensor` argument provides the encoded design vector as the model's conditioning input. Gradient-based optimization also requires a `backward()` implementation:

```python
def __call__(self, design, design_tensor=None, **kwargs):
    self.get_runtime_model()
    predictions = self.model(design_tensor)
    return surrogate_predictions_to_dict(predictions, self.keys, n=1)

def backward(self, design, grad_output=None, design_tensor=None):
    if grad_output is None:
        return
    keys = list(grad_output.keys())
    results = self(design, design_tensor=design_tensor)
    tensors = [results[k] for k in keys]
    grad_tensors = [grad_output[k] for k in grad_output]
    torch.autograd.backward(tensors, grad_tensors)
```

!!! note "Training data"

    Workflow surrogates read the results recorded from real workflow runs. `WorkflowSurrogate` registers the records file (`DefaultPaths.TRAINING_RECORDS`) automatically as an `Input` — during a training session it resolves to the data collected from the strategy's training batch.

## Module Surrogates

A module surrogate replaces a single module inside the workflow. Instead of predicting objective values, it predicts the module's *output files* from its inputs, so all downstream modules keep working unchanged.

The connection between module and surrogate is the **surrogate contract**. The contract describes the module's inputs and outputs, plus any additional conditioning the surrogate needs beyond the design parameters — for example a particle type or sampled particle energies. The contract captures exactly this domain knowledge, while the surrogate itself stays a generic model that can be trained and evaluated without knowing anything about the underlying process. The module exposes the contract as a property:

```python
from dsbo.utils.io import LazyShape
from dsbo.surrogates.module_surrogate import ModuleSurrogateContract

class MySimulation(Module):
    def __init__(self):
        super().__init__()
        self.output_file = Workdir.register_file(DefaultPaths.SIMULATION, "showers.h5")
        self.target_file = Workdir.register_file(DefaultPaths.SIMULATION, "targets.h5")

    @property
    def surrogate_contract(self) -> ModuleSurrogateContract:
        return ModuleSurrogateContract(
            output_dim=LazyShape(self.output_file),
            output_samples=...,
            conditional_dim=LazyShape(self.target_file),
            output_conditional_fn=lambda design: ...,
            data_file=self.output_file,
            conditional_file=self.target_file,
        )
```

The contract tells the surrogate:

- `output_dim` / `conditional_dim` — the shape of the module's output and conditioning tensors. `LazyShape` infers them from existing files, avoiding hard-coded dimensions.
- `output_samples` — a required positive sample count declared by the contract. Custom module surrogates can use it to control how many samples they generate per evaluation.
- `output_conditional_fn` — how to build the additional conditioning tensor (e.g. particle type, sampled energies) that the model is trained on; defaults to the design values. Designs containing non-numeric parameters require an explicit implementation.
- `data_file` / `conditional_file` — where training data comes from and where predictions are written. Both are required to train the surrogate.

The module's `patch()` method attaches the surrogate and returns the module, so it composes with the workflow DSL:

```python
from dsbo.surrogates.module_surrogate import DiffusersModuleSurrogate

sim = MySimulation().patch(DiffusersModuleSurrogate(MyStrategy()))
wf = Workflow(sim >> reco)
```

At evaluation time the surrogate writes its prediction to the contract's data file; downstream modules read it exactly as if the real module had produced it. The shipped module surrogate writes only primal predictions and provides no backward pass, so gradients do not flow through the surrogate itself.

During a training session, the real module and its upstream modules run on the strategy's training candidates — the module's output and conditioning files then hold the data the surrogate is trained on.

!!! warning "No batched evaluation"

    Workflows with module surrogates currently support single-candidate evaluation only. Passing more than one candidate to `run_batch()` raises a `NotImplementedError`.

Custom module surrogates extend `ModuleSurrogate` with a `train` (or `pretrain`/`finetune`) lifecycle task that fits a model to the contract's files, and a `__call__` method that writes the prediction into the contract's data file via `contract.write_output(design, output_tensor)`.

## Pre-Built Surrogates

dsbo ships diffusion-based surrogates with built-in model architectures and training routines:

| Surrogate | Kind | Configure via |
|---|---|---|
| `DefaultWorkflowSurrogate` | Workflow surrogate | Constructor arguments (architecture, epochs, batch size, sampling steps, ...) |
| `DiffusersModuleSurrogate` | Module surrogate | The module's `surrogate_contract` |

Both learn a conditional generative model over the outputs and produce *sampled* predictions rather than point estimates. Workflow surrogates expose `sample_count` to control the number of samples per prediction; the shipped module surrogate generates one prediction per row of the conditioning tensor returned by `output_conditional_fn`.

```python
from dsbo.surrogates.workflow_surrogate import DefaultWorkflowSurrogate

wf = Workflow(sim >> reco)
wf.patch(DefaultWorkflowSurrogate(MyStrategy(), max_epochs=100, sample_count=32))
```

## Advanced Training and Validation

For shared reconstruction training, see [Batch-Fused Workflow Stages](batch-fusion.md).

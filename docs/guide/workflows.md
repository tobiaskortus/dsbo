# Building Workflows

A **workflow** is a directed acyclic graph (DAG) of module tasks. Dependencies are declared with the operator DSL below; from the graph DSBO computes a topological task order. Local execution is sequential; with an active distributed worker, independent tasks can run concurrently on the compute resources declared on their modules (see [Parallelization with Dask](parallelization.md)).

## Defining a Workflow

A `Workflow` wraps a task graph — a single module, or any composition produced by the operators below:

```python
from dsbo.workflow import Workflow

wf = Workflow(module_a >> module_b)
```

Evaluate it with `run()`: the graph executes once for the given candidate and the metric dictionaries of all tasks are merged into one result. A result can be used as an optimization objective when its contributing modules implement the required derivatives; returning a diagnostic metric does not automatically supply its backward implementation:

```python
result = wf.run(candidate)

result["energy_resolution"]   # named metric of one task
```

## Batch Evaluation

`run_batch()` evaluates a sequence of candidates and returns a `ResultsBatch`, available from `dsbo.workflow`. This list-like object contains one metric dictionary per candidate, in the same order as the input candidates. Each dictionary combines the named results returned by that candidate's workflow tasks.

| Access | Result |
|---|---|
| `len(results)` | Number of candidate results |
| `results[i]` | Metric dictionary for candidate `i` |
| `for row in results` | Iterate over the per-candidate dictionaries |
| `results["metric"]` | Collect a named metric across candidates; tensor values are concatenated, while non-tensor values are returned as a list |
| `results.keys` | Metric names from the first candidate's dictionary; empty for an empty batch |
| `results.tensor` | For scalar metrics, a reporting matrix with candidates as rows and metrics as columns, using each dictionary's value order; this creates a new tensor without the original gradient connections |

Use explicit stacking for scalar objectives to preserve their autograd connections:

```python
import torch

results = wf.run_batch([candidate_1, candidate_2, candidate_3])

results[0]  # metric dictionary of the first candidate
losses = torch.stack([row["energy_resolution"] for row in results])
# losses.mean().backward() when candidates and modules support differentiation
```

![Sequential pipeline: ModuleA feeds ModuleB](../img/workflow-sequential.png)

The graph is stored as a `networkx.DiGraph` and validated at construction — cycles (`a >> a`) and duplicate module instances are rejected before any task runs.

## Composing Modules

Three operators cover the common patterns:

| Operator | Effect |
|---|---|
| <code>a >> b</code> | adds the dependency edge <code>a → b</code>: <code>b</code> runs after <code>a</code> |
| <code>a &#124; b</code> | <code>a</code> and <code>b</code> run independently; results are merged |
| <code>a * n</code> | replicates <code>a</code> into <code>n</code> tasks <code>a[0] … a[n-1]</code>, each with its own replica index |

Use `|` for independent evaluations whose metrics should end up in one result dict:

```python
wf = Workflow(module_a | module_b)
```

![Parallel evaluations: ModuleA and ModuleB run independently and their metrics are merged](../img/workflow-parallel.png)

Chaining composes naturally — every operator returns a dependency dictionary that the next operator consumes. Branches merge the same way, so a stage can feed several consumers:

```python
wf = Workflow(
    (module_a >> module_b)          # ModuleA feeds ModuleB ...
    | (module_a >> module_c)        # ... and, independently, ModuleC
)
```

![Branching: ModuleA feeds both ModuleB and ModuleC](../img/workflow-branching.png)

A replicated stage feeding a single consumer is a **fan-in** — every replica feeds the single consumer:

```python
wf = Workflow(module_a * 2 >> module_b)
```

![Fan-in: two ModuleA replicas feed one ModuleB stage](../img/workflow-fanin.png)

Replicas paired with replicas are matched **one-to-one by replica index**. This runs three independent ModuleA-and-ModuleB pipelines — no cross-connections:

```python
CPU_CONFIG = SLURMWorkerSpec(processes=5, cores=5, walltime="48:00:00", memory="48GB", partition="epyc-256")
GPU_CONFIG = SLURMWorkerSpec(processes=1, cores=4, gpu_resources="gpu:1", memory="16GB", walltime="48:00:00")

wf = Workflow(module_a.to(CPU_CONFIG) * 3 >> module_b.to(GPU_CONFIG) * 3)
```

![One-to-one pairing: each ModuleA replica feeds its own ModuleB replica](../img/workflow-pairing.png)

!!! note "Fan-in versus pairing"

    `module_a * 2 >> module_b` connects *every* replica to `module_b`. `module_a * 3 >> module_b * 3` pairs replica *i* with replica *i*.

!!! warning "Replicated modules must be stateless"

    Replicas are shallow copies that share mutable sub-objects and differ only in `.i`. Never rely on instance state surviving across replicas — treat modules as stateless.

## Per-Replica Subgraphs

To give every replica its own *multi-stage* pipeline, bind a subgraph to `dynamic()`. The subgraph is copied once per left-hand replica and wired by index:

```python
from dsbo.modules.dsl import dynamic

per_instance = (module_b.train >> module_b.validate) * dynamic()
wf = Workflow(module_a * 3 >> per_instance)
```

![Per-replica subgraphs: each ModuleA replica drives its own train and validate chain of ModuleB](../img/workflow-dynamic.png)

This scales to pipelines of any depth — the expression stays the same no matter how many stages follow each replica. Note that the stages of one pipeline share its single `module_a` run. When the stages need **individual runs** instead, replicate them and rely on the one-to-one pairing — three pipelines with a training and a validation stage need six simulations:

```python
wf = Workflow(module_a * 6 >> (module_b.train | module_b.validate) * 3)
```

The replicas are paired by index in stage order, so every `train` and `validate` stage consumes its own `module_a` run. The trade-off: the pairing form does not order `validate` after `train` — stages that depend on another stage's outputs (a trained model, for instance) should be chained in a bound subgraph as above, or given per-stage resources and explicit file dependencies.

## Dynamic Submodules

Large modules are easier to schedule when their internal steps are separate tasks. `@dynamic_submodule` turns a method into a first-class graph node that shares the parent module's files, configuration, and state:

```python
from dsbo.autograd import train_op
from dsbo.modules import Module, dynamic_submodule

class ModuleB(Module):
    @train_op
    @dynamic_submodule
    def train(self, design, **kwargs):
        model = fit_model(self.training_data())
        torch.save(model.state_dict(), self.model_file())
        return {}

    @dynamic_submodule
    def validate(self, design, **kwargs) -> dict[str, torch.Tensor]:
        model = load_model(self.model_file())
        return evaluate(model, self.validation_data())
```

The decorated methods are ordinary graph nodes, so they compose with the same operators — including resource assignment per stage. The stages above each declare their own data inputs (`self.training_data()` for training, `self.validation_data()` for validation), so **each stage needs its own `module_a` run** — one producing the training data, one producing the validation data — and `validate` additionally depends on `train` for the trained model:

```python
wf = Workflow(
    (module_a.replicate(0) >> module_b.train)       # ModuleA run for the training data
    | (module_a.replicate(1) >> module_b.validate)  # separate ModuleA run for validation data
    | (module_b.train >> module_b.validate)         # validate loads the model train wrote
)
```

![Dynamic submodules as explicit stages: individual ModuleA runs feed train and validate, and validate loads the model written by train](../img/workflow-submodules.png)

At runtime the submodules delegate to their parent: `module_b.train` runs the `train` method on the parent instance, so anything stored on `self` during `train` is visible to `validate` — provided both run in the same process. Like all modules, treat them as stateless across tasks and pass data through files instead (as the `model_file()` above does; see [I/O and Autograd](io.md)).

A few rules of thumb:

- **Split stages that need different resources.** Training on GPU while post-processing runs on CPU is only possible if they are separate tasks.
- **Split stages that should be re-run independently**, e.g. re-validating with a cached model without retraining.
- **Wrap training loops with `@train_op`** (stacked above `@dynamic_submodule`), as the `train` stage above does.

`TrainableModule` — the base class of all surrogates — uses the same mechanism for its lifecycle: its `train`, `pretrain`, and `finetune` stages are dynamic submodules, which is why `SurrogateBase` can be trained through the workflow (see [Writing Surrogates](surrogates.md)).

### Training Loops with `train_op`

Workflow execution controls PyTorch's gradient context, but a model's training loop needs its own gradients to update model parameters. Decorate a task containing such a loop with `@train_op`, imported from `dsbo.autograd`, and stack it above `@dynamic_submodule` as in the example above.

For the duration of the training call, `train_op` enables PyTorch reverse-mode gradient tracking, even if the surrounding workflow has disabled it. This allows the training loss to call `backward()` and the optimizer to update the model parameters. It also temporarily exits active forward-mode AD levels so that training does not run inside the workflow's tangent-propagation context. The previous gradient mode and forward-mode levels are restored when the call finishes, including when it raises an exception.

`train_op` controls the local training context; it does not supply a workflow `backward()` implementation or differentiate through the optimizer steps. It also does not exclude the task from the workflow's reverse pass. Use `with_no_grad` on a training branch when that branch should be excluded from reverse execution, as shown below. The two mechanisms serve different purposes: `with_no_grad` controls workflow-level reverse execution, while `train_op` enables the gradients needed inside the training task.

## Excluding Stages from Gradient Tracking

Stages that must not participate in the reverse pass, such as a training-data branch, are marked with `with_no_grad`. It accepts a single module or a whole composition and flags every task in it. Applied to the training path below, the training simulation and `train` stage are excluded from reverse execution. Validation differentiates its supported inputs using a fixed trained model; the model-file dependency does not differentiate the training procedure:

```python
from dsbo.modules import with_no_grad

wf = Workflow(
    with_no_grad(module_a.replicate(0) >> module_b.train)  # training path: no gradients
    | (module_a.replicate(1) >> module_b.validate)
    | (module_b.train >> module_b.validate)
)
```

![Dynamic submodules workflow with the training path excluded from gradient tracking: the ModuleA replica producing the training data and the train stage carry the no-grad flag](../img/workflow-no-grad.png)

This flag is not a forward-mode stop-gradient: forward AD can still propagate tangents through a flagged module unless that implementation suppresses them. See [Reverse and Forward Gradient Flow](gradient-flow.md) for mode selection and gradient propagation.

## Batch-Fused Stages

Batch evaluation runs every stage once per candidate. Stages that operate on the batch as a whole — a model trained on all candidates' data, for instance — can instead be wrapped with `batch_fuse`. Such a stage runs **once per batch**: the inputs of all candidates are reduced into a single file (see [Combining Replica Outputs](io.md#combining-replica-outputs)), and the stage consumes them in a single call:

```python
from dsbo.modules import batch_fuse

wf = Workflow(module_a >> batch_fuse(module_b))
```

The wrapped module must register at least one `Input`, and its returned metrics are discarded; results flow through files that the framework makes visible to each candidate again. Batch-fused stages are excluded from reverse execution. [Batch-Fused Workflow Stages](batch-fusion.md) explains first-candidate arguments, data ordering, shared outputs, and the forward-mode limitations.

## Resource Annotation

Every module can carry a worker specification. Assign it with `.to()`, which returns the module and therefore composes with the DSL:

```python
from dsbo.worker.spec import SLURMWorkerSpec

CPU_CONFIG = SLURMWorkerSpec(processes=5, cores=5, walltime="48:00:00", memory="48GB", partition="epyc-256")
GPU_CONFIG = SLURMWorkerSpec(processes=1, cores=4, gpu_resources="gpu:1", memory="16GB", walltime="48:00:00")

wf = Workflow(module_a.to(CPU_CONFIG) >> module_b.to(GPU_CONFIG))
```

For local debugging, run the same workflow without any worker — DSBO executes it in-process:

```python
result = wf.run(candidate)   # no worker required
```

Distributed execution with Dask is covered in [Parallelization with Dask](parallelization.md) — the same workflow runs unchanged, with workers derived from these annotations.

## Inspecting Workflows

The task graph is a plain `networkx.DiGraph`, so it can be inspected directly:

```python
wf.task_graph
```

To render it as an image, call `visualize()` — each node is colored by its assigned resources:

```python
wf.visualize("workflow.png")
```

## Results and Failure Handling

Keep metric names and per-metric shapes consistent across a differentiable batch. Reverse-mode result packing uses the first candidate's result schema; missing keys or incompatible shapes in later results are not automatically padded or converted into failed-row markers. Duplicate metric names across tasks are errors, not namespaces.

`run()` and `run_batch()` accept `catch=(SomeException,)` to catch selected forward task exceptions. This is not a retry or recovery policy. Local execution can continue to downstream tasks after a caught producer failure, where missing files may cause further errors. Distributed result gathering can omit failed task results. Partial results can therefore fail later during aggregation or differentiation. The custom reverse call does not preserve the original forward `catch` policy.

[Logging Metrics](metrics.md) distinguishes task traces from objective histories.

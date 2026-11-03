# Surrogate-Assisted Optimization

This example extends [Gradient-Based Optimization](gradient-based-optimization.md) by replacing repeated full workflow evaluations with a learned approximation. The optimizer still operates on the candidate tensor; the surrogate strategy decides when predictions can be used and when new real evaluations are needed.

Use the imports, design space, `ModuleA`, `ModuleB`, and `TARGET` definitions from that example's **Design Space**, **Modules**, and **Workflow** sections. The direct optimization loop need not be rerun. Execute the blocks below in the same script or session after that setup. See [Writing Surrogates](surrogates.md) for the general interfaces and training lifecycle.

## Trust-Region Strategy

The strategy samples training candidates around the current design and allows predictions while the candidate remains near that training center:

```python
import torch
from dsbo.parameters import Candidate, CandidateBatch, ParameterSpace
from dsbo.surrogates import SurrogateStrategy


class LocalTrustRegion(SurrogateStrategy):
    def __init__(self, num_candidates=16, radius=0.2, spread=0.15):
        self.num_candidates = num_candidates
        self.radius = radius
        self.spread = spread
        self.center = None

    def is_in_trust_region(self, candidate: Candidate) -> bool:
        if self.center is None:
            return False
        return bool((candidate.tensor - self.center).norm() < self.radius)

    def generate_candidates(self, candidate, space, finetune=False):
        self.center = candidate.tensor.detach().clone()
        return CandidateBatch([
            Candidate(space, self.center + self.spread * torch.randn_like(self.center))
            for _ in range(self.num_candidates)
        ])
```

These Gaussian training points are not clipped to domain bounds. A strategy for a constrained problem must also enforce the problem's sampling and feasibility requirements.

## Linear Workflow Surrogate

A local linear model is sufficient to illustrate the interface. The base class provides recorded workflow results and normalization statistics. The training task fits the model and saves it; prediction restores the published model and maps its output back to the workflow's result names.

The training method uses a [dynamic submodule](module-templates-and-dynamic-submodules.md#module-with-dynamic-submodules), so it can be scheduled through the same runtime as the real workflow:

```python
from dsbo.modules import dynamic_submodule
from dsbo.surrogates.workflow_surrogate import WorkflowSurrogate, surrogate_predictions_to_dict


class LinearWorkflowSurrogate(WorkflowSurrogate):
    def __init__(self, strategy):
        super().__init__(strategy=strategy)
        self.val_split = 0.2
        self.seed = 0
        self.batch_size = 32
        self.num_workers = 0
        self._weight = None
        self._bias = None
        self.surrogate = None

    @dynamic_submodule
    def train(self, design, candidates):
        train_loader, _ = self.get_train_val_loaders(self.data_path(), candidates)
        try:
            designs, results = [], []
            for y, x in train_loader:
                designs.append(x.reshape(x.shape[0], -1))
                results.append(y.reshape(y.shape[0], -1))
            x = torch.cat(designs)
            y = torch.cat(results)

            x = torch.cat([x, torch.ones(len(x), 1)], dim=1)
            solution = torch.linalg.lstsq(x, y).solution
            self._weight = solution[:-1].T.contiguous()
            self._bias = solution[-1]
            torch.save(self.state_dict(), self.model_file_path())
        finally:
            if getattr(self, "_records_dataset", None) is not None:
                self._records_dataset.close()
                self._records_dataset = None
        return {}

    def state_dict(self):
        if self._weight is None:
            return {}
        return {
            "weight": self._weight,
            "bias": self._bias,
            "mean_x": self._mean_x,
            "std_x": self._std_x,
            "mean_y": self._mean_y,
            "std_y": self._std_y,
            "keys": self.keys,
        }

    def from_state_dict(self, state):
        self._weight = state["weight"]
        self._bias = state["bias"]
        self._mean_x = state.get("mean_x")
        self._std_x = state.get("std_x")
        self._mean_y = state.get("mean_y")
        self._std_y = state.get("std_y")
        self.keys = state["keys"]
        self.surrogate = (self._weight, self._bias)

    def __call__(self, design, design_tensor=None, **kwargs):
        self.get_runtime_model()
        x = (design_tensor.float() - self._mean_x) / self._std_x
        y = x @ self._weight.T + self._bias
        y = y * self._std_y + self._mean_y
        return surrogate_predictions_to_dict(y.unsqueeze(0), self.keys, 1)
```

## Optimization Loop

Compared with direct optimization, this loop adds a patched workflow, a persistent working directory, and study storage for the published model. The optimizer and its backward/update steps are unchanged:

```python
from dsbo.storage import use_storage
from dsbo.storage.sqlmodel import StudyStorage

torch.manual_seed(0)

a2 = ModuleA()
b2 = ModuleB(a2.out.path)
wf_s = Workflow(a2 >> b2)

with Workdir.create("./example_surrogate", keep=True):
    with use_storage(StudyStorage.create("./example_surrogate/study.db")):
        wf_s.patch(LinearWorkflowSurrogate(LocalTrustRegion()))
        candidate = Candidate.from_values(
            space, {"gain": 0.3, "bias": -1.5}, requires_grad=True
        )
        optimizer = torch.optim.Adam([candidate.tensor], lr=0.01)

        for step in range(200):
            result = wf_s.run(candidate)
            loss = result["loss"].sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 20 == 0:
                values = candidate.detach().values
                print(f"step {step:3d}  loss {loss.item():.6f}  "
                      f"gain {float(values['gain']):.3f}  bias {float(values['bias']):.3f}")

values = candidate.detach().values
response = float(values["gain"]) * torch.tanh(torch.tensor(float(values["bias"])))
print(f"final: response = {response:.4f} (target {TARGET})")
```

## Interpreting the Results

The first `run()` evaluates the real workflow on a training batch, fits the linear surrogate to the recorded results, and saves the fitted model as an artifact. Subsequent optimization steps use this model while the candidate remains inside the strategy's trust region. Leaving that region triggers another training session around the current candidate.

The printed `loss` is the surrogate's estimate of the objective. Unlike the squared-error objective, the linear model is not constrained to nonnegative values, so a negative prediction indicates approximation error rather than a better solution. The final line evaluates the original response expression at the optimized parameters, providing a direct comparison with `TARGET` independent of the surrogate.

The computational benefit depends on how many surrogate evaluations occur between training sessions and how expensive the real workflow is. This small example demonstrates the integration; it does not establish a speedup over direct gradient-based optimization.

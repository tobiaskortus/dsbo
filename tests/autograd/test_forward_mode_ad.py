import torch
import numpy as np
import torch.autograd.forward_ad as fw
from tempfile import TemporaryDirectory

from dsbo.utils.io.workdir import Workdir
from dsbo.worker.dask import LocalDaskWorker, WorkerContext

def test_forward_mode_ad_local(space, workflow, sampler):
    with fw.dual_level():
        space.register_sampler(sampler)
        candidate = space.sample(requires_grad=True)
        
        # Calculate d(loss)/dx
        with TemporaryDirectory() as tempdir, Workdir.create(tempdir, keep=True, error_if_exists=False):
                candidate = candidate.set_tangent(torch.tensor([1.0, 0.0]))
                loss = workflow.run(candidate)

                x0, y0 = candidate.tensor.tolist()
                expected_dx = float(y0 * np.exp(x0*y0) * (np.cos(x0*y0) + np.sin(x0*y0)))

                assert np.isclose(fw.unpack_dual(loss["final"]).tangent.item(), expected_dx, atol=1e-5), f"Expected d(loss)/dx to be {expected_dx:.6f}, but got {candidate.tensor.grad[0].item():.6f}"

        # Calculate d(loss)/dy
        with TemporaryDirectory() as tempdir, Workdir.create(tempdir, keep=True, error_if_exists=False):
                candidate = candidate.set_tangent(torch.tensor([0.0, 1.0]))
                loss = workflow.run(candidate)

                x0, y0 = candidate.tensor.tolist()
                expected_dy = float(x0 * np.exp(x0*y0) * (np.cos(x0*y0) + np.sin(x0*y0)))

                assert np.isclose(fw.unpack_dual(loss["final"]).tangent.item(), expected_dy, atol=1e-5), f"Expected d(loss)/dy to be {expected_dy:.6f}, but got {candidate.tensor.grad[1].item():.6f}"

        
def test_forward_mode_ad_distributed(space, workflow, sampler):
    with fw.dual_level():
        space.register_sampler(sampler)
        candidate = space.sample(requires_grad=True)

        with LocalDaskWorker() as worker, WorkerContext.use_worker(worker):
            # Calculate d(loss)/dx
            with TemporaryDirectory() as tempdir, Workdir.create(tempdir, keep=True, error_if_exists=False):
                candidate = candidate.set_tangent(torch.tensor([1.0, 0.0]))
                loss = workflow.run(candidate)

                x0, y0 = candidate.tensor.tolist()
                expected_dx = float(y0 * np.exp(x0*y0) * (np.cos(x0*y0) + np.sin(x0*y0)))

                assert np.isclose(fw.unpack_dual(loss["final"]).tangent.item(), expected_dx, atol=1e-5), f"Expected d(loss)/dx to be {expected_dx:.6f}, but got {fw.unpack_dual(loss['final']).tangent.item():.6f}"

            # Calculate d(loss)/dy
            with TemporaryDirectory() as tempdir, Workdir.create(tempdir, keep=True, error_if_exists=False):
                candidate = candidate.set_tangent(torch.tensor([0.0, 1.0]))
                loss = workflow.run(candidate)

                x0, y0 = candidate.tensor.tolist()
                expected_dy = float(x0 * np.exp(x0*y0) * (np.cos(x0*y0) + np.sin(x0*y0)))

                assert np.isclose(fw.unpack_dual(loss["final"]).tangent.item(), expected_dy, atol=1e-5), f"Expected d(loss)/dy to be {expected_dy:.6f}, but got {fw.unpack_dual(loss['final']).tangent.item():.6f}"
      
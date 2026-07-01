import torch
import pytest
import numpy as np
from tempfile import TemporaryDirectory

from dsbo.utils.io.workdir import Workdir
from dsbo.worker.dask import LocalDaskWorker, WorkerContext


def test_reverse_mode_ad_local(space, workflow, sampler):
    space.register_sampler(sampler)
    candidate = space.sample(requires_grad=True)
    
    with TemporaryDirectory() as tempdir, Workdir.create(tempdir, keep=True, error_if_exists=False):
        loss = workflow.run(candidate)
        loss["final"].backward()

        x0, y0 = candidate.tensor.tolist()
        expected_dx = float(y0 * np.exp(x0*y0) * (np.cos(x0*y0) + np.sin(x0*y0)))
        expected_dy = float(x0 * np.exp(x0*y0) * (np.cos(x0*y0) + np.sin(x0*y0)))

        assert np.isclose(candidate.tensor.grad[0].item(), expected_dx, atol=1e-5), f"Expected d(loss)/dx to be {expected_dx:.6f}, but got {candidate.tensor.grad[0].item():.6f}"
        assert np.isclose(candidate.tensor.grad[1].item(), expected_dy, atol=1e-5), f"Expected d(loss)/dy to be {expected_dy:.6f}, but got {candidate.tensor.grad[1].item():.6f}"



def test_reverse_mode_ad_distributed(space, workflow, sampler):
    space.register_sampler(sampler)
    candidate = space.sample(requires_grad=True)
    
    with LocalDaskWorker() as worker, WorkerContext.use_worker(worker):
        with TemporaryDirectory() as tmpdir, Workdir.create(tmpdir, keep=True, error_if_exists=False):
            loss = workflow.run(candidate)
            loss["final"].backward()

        x0, y0 = candidate.tensor.tolist()
        expected_dx = float(y0 * np.exp(x0*y0) * (np.cos(x0*y0) + np.sin(x0*y0)))
        expected_dy = float(x0 * np.exp(x0*y0) * (np.cos(x0*y0) + np.sin(x0*y0)))

        assert np.isclose(candidate.tensor.grad[0].item(), expected_dx, atol=1e-5), f"Expected d(loss)/dx to be {expected_dx:.6f}, but got {candidate.tensor.grad[0].item():.6f}"
        assert np.isclose(candidate.tensor.grad[1].item(), expected_dy, atol=1e-5), f"Expected d(loss)/dy to be {expected_dy:.6f}, but got {candidate.tensor.grad[1].item():.6f}"
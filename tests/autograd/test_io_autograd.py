import os
from unittest.mock import patch
from tempfile import TemporaryDirectory

import numpy as np
import pytest
import torch
import torch.autograd.forward_ad as fw

from dsbo.utils.io.autograd import NumpyFile, InMemoryFile

def test_numpy_file_primal_only():
    #dual_level==-1 and no_grad
    with torch.no_grad():
        x = torch.tensor([1.0, 2.0])
        x = x * 3

        with TemporaryDirectory() as tempdir:
            file_path = os.path.join(tempdir, "x.npz")
            NumpyFile.write(file_path, x)
            x_loaded = NumpyFile.open(file_path, "cpu")

            torch.testing.assert_close(x_loaded, torch.tensor([3.0, 6.0]))


def test_numpy_file_forward_ad_tangent():
    with fw.dual_level():
        x = fw.make_dual(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 0.0]))
        x = x * 3

        with TemporaryDirectory() as tempdir:
            file_path = os.path.join(tempdir, "x.npz")
            NumpyFile.write(file_path, x)
            x_loaded = NumpyFile.open(file_path, "cpu")
            tangent = fw.unpack_dual(x_loaded).tangent
            
            torch.testing.assert_close(x_loaded, torch.tensor([3.0, 6.0]))
            torch.testing.assert_close(tangent, torch.tensor([3.0, 0.0]))


def test_numpy_file_backward_gradient():
    with TemporaryDirectory() as tempdir:
        file_path = os.path.join(tempdir, "x.npz")

        primal = torch.tensor([2.0, 3.0])
        NumpyFile.write(file_path, primal)

        with torch.enable_grad():
            x = NumpyFile.open(file_path, "cpu")
            y = (x * x).sum()

            # Check that the autograd hook was registered during open()
            assert file_path in NumpyFile.handles
            
            y.backward()

        assert os.path.exists(os.path.join(tempdir, "x_gradient.npz"))
        grad = NumpyFile.read_backward_gradient(file_path)
        torch.testing.assert_close(grad, 2.0 * primal)


def test_inmemory_file_primal_only():
    with torch.no_grad():
        x = torch.tensor([1.0, 2.0])
        x = x * 3
        file_path = "inmem_x"

        InMemoryFile._storage.clear()
        InMemoryFile.write(file_path, x)

        x_loaded = InMemoryFile.open(file_path, "cpu")

        torch.testing.assert_close(x_loaded, torch.tensor([3.0, 6.0]))


def test_inmemory_file_forward_ad_tangent():
    with fw.dual_level():
        x = fw.make_dual(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 0.0]))
        x = x * 3
        file_path = "inmem_x"

        InMemoryFile._storage.clear()
        InMemoryFile.write(file_path, x)

        x_loaded = InMemoryFile.open(file_path, "cpu")
        tangent = fw.unpack_dual(x_loaded).tangent
        
        torch.testing.assert_close(x_loaded, torch.tensor([3.0, 6.0]))
        torch.testing.assert_close(tangent, torch.tensor([3.0, 0.0]))


def test_inmemory_file_backward_gradient():
    file_path = "inmem_x"
    InMemoryFile._storage.clear()
    primal = torch.tensor([2.0, 3.0])
    InMemoryFile.write(file_path, primal)

    with torch.enable_grad():
        x = InMemoryFile.open(file_path, "cpu")
        y = (x * x).sum()

        assert file_path in InMemoryFile._storage

        y.backward()

    grad = InMemoryFile.read_backward_gradient(file_path)
    torch.testing.assert_close(grad, 2.0 * primal)

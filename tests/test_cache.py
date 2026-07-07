import os
import json
import torch
import pytest
import shutil
import hashlib
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from dsbo.workflow import Workflow
from dsbo.utils.random import seed_all, get_random_engine
from dsbo.utils.cache import (
    CachePolicy,
    CacheConfig,
    module_code,
    compute_workflow_hash,
    compute_candidate_hash,
    compute_candidate_hash,
    write_npz,
    load_npz,
    save_cache_entry,
    load_cache_entry,
    check_cache_entry,
    WorkflowExecutionCache,
    CACHE_DIR,
    MANIFEST_NAME,
    RESULTS_NAME,
)
from dsbo.modules import Module
from dsbo.parameters import ParameterSpace, Vector, Candidate, CandidateBatch
from dsbo.surrogates.graph_transformations import GraphMode



class SimpleModule(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, design, **kwargs):
        return {"out": design["x"] * 2.0}


class AnotherModule(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, design, **kwargs):
        return {"out": design["x"] + 1.0}


@pytest.fixture
def simple_space():
    return ParameterSpace(x=Vector(0, 10, 1))


@pytest.fixture
def candidate(simple_space):
    t = torch.tensor([5.0])
    return Candidate(simple_space, t)


@pytest.fixture
def candidate_batch(simple_space):
    t = torch.stack([torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])])
    return CandidateBatch([Candidate(simple_space, t[i]) for i in range(t.shape[0])])


@pytest.fixture
def results():
    return {"out": torch.tensor(10.0)}


@pytest.fixture
def results_with_tangent():
    import torch.autograd.forward_ad as fw
    primal = torch.tensor(10.0)
    tangent = torch.tensor(2.0)
    with fw.dual_level():
        yield {"out": fw.make_dual(primal, tangent)}


@pytest.fixture
def temp_dir():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(d, ignore_errors=True)

def test_cache_everything_always_caches():
    cfg = CacheConfig(cache_policy=CachePolicy.CACHE_EVERYTHING)
    assert cfg.should_cache(GraphMode.DEFAULT)
    assert cfg.should_cache(GraphMode.MODULE_SURROGATE_EVAL)
    assert cfg.should_cache(GraphMode.MODULE_SURROGATE_TRAIN)

def test_retrain_surrogates_only_default():
    cfg = CacheConfig(cache_policy=CachePolicy.RETRAIN_SURROGATES)
    assert cfg.should_cache(GraphMode.DEFAULT)
    assert cfg.should_cache(GraphMode.MODULE_SURROGATE_TRAIN)
    assert not cfg.should_cache(GraphMode.MODULE_SURROGATE_EVAL)

def test_always_execute_never_caches():
    cfg = CacheConfig(cache_policy=CachePolicy.ALWAYS_EXECUTE)
    assert not cfg.should_cache(GraphMode.DEFAULT)
    assert not cfg.should_cache(GraphMode.MODULE_SURROGATE_EVAL)
    assert not cfg.should_cache(GraphMode.MODULE_SURROGATE_TRAIN)

def test_default_policy_is_cache_everything():
    cfg = CacheConfig()
    assert cfg.cache_policy == CachePolicy.CACHE_EVERYTHING

def test_module_code_metadata():
    m = SimpleModule()
    code = module_code(m)
    assert "test_cache.SimpleModule" in code
    assert "seed=None" in code
    assert "no_grad=False" in code

def test_different_classes_differ():
    code_a = module_code(SimpleModule())
    code_b = module_code(AnotherModule())
    assert code_a != code_b

def test_same_class_same_code():
    code_a = module_code(SimpleModule())
    code_b = module_code(SimpleModule())
    assert code_a == code_b

def test_seed_changes_code():
    m1 = SimpleModule()
    m1._seed = 42
    m2 = SimpleModule()
    m2._seed = 43
    assert module_code(m1) != module_code(m2)

def test_no_grad_changes_code():
    m1 = SimpleModule()
    m1._no_grad_op = False
    m2 = SimpleModule()
    m2._no_grad_op = True
    assert module_code(m1) != module_code(m2)

def test_same_graph_same_hash():
    m1 = SimpleModule()
    m2 = AnotherModule()
    wf_a = Workflow(m1 >> m2)
    wf_b = Workflow(m1 >> m2)
    assert compute_workflow_hash(wf_a) == compute_workflow_hash(wf_b)

def test_different_graph_different_hash():
    m1 = SimpleModule()
    m2 = AnotherModule()
    h_a = compute_workflow_hash(Workflow(m1 >> m2))
    h_b = compute_workflow_hash(Workflow(m2 >> m1))
    assert h_a != h_b

def test_hash_is_deterministic():
    m1 = SimpleModule()
    m2 = AnotherModule()
    wf = Workflow(m1 >> m2)
    h1 = compute_workflow_hash(wf)
    h2 = compute_workflow_hash(wf)
    assert h1 == h2

def test_same_candidate_same_hash(simple_space):
    c1 = Candidate(simple_space, torch.tensor([5.0]))
    c2 = Candidate(simple_space, torch.tensor([5.0]))
    assert compute_candidate_hash(c1) == compute_candidate_hash(c2)

def test_different_candidate_different_hash(simple_space):
    c1 = Candidate(simple_space, torch.tensor([5.0]))
    c2 = Candidate(simple_space, torch.tensor([7.0]))
    assert compute_candidate_hash(c1) != compute_candidate_hash(c2)

def test_hash_is_deterministic(simple_space):
    c = Candidate(simple_space, torch.tensor([5.0]))
    assert compute_candidate_hash(c) == compute_candidate_hash(c)

def test_write_load_scalar_results(temp_dir, results):
    tangent_keys = write_npz(str(temp_dir), results)
    assert tangent_keys == []
    manifest = {"result_keys": ["out"], "tangent_keys": []}
    loaded = load_npz(str(temp_dir), manifest)
    assert loaded is not None
    assert torch.allclose(loaded["out"], results["out"])

def test_write_load_with_tangent(temp_dir, results_with_tangent):
    import torch.autograd.forward_ad as fw
    tangent_keys = write_npz(str(temp_dir), results_with_tangent)
    assert "out" in tangent_keys
    manifest = {"result_keys": ["out"], "tangent_keys": tangent_keys}
    loaded = load_npz(str(temp_dir), manifest)
    assert loaded is not None
    primal, tan = fw.unpack_dual(loaded["out"])
    assert torch.allclose(primal, torch.tensor(10.0))
    assert torch.allclose(tan, torch.tensor(2.0))

def test_load_missing_file(temp_dir):
    manifest = {"result_keys": ["out"], "tangent_keys": []}
    assert load_npz(str(temp_dir), manifest) is None

def test_check_missing_entry(temp_dir):
    assert not check_cache_entry(str(temp_dir), "nosuch", "nosuch")

def test_load_missing_entry(temp_dir):
    assert load_cache_entry(str(temp_dir), "nosuch", "nosuch") is None

def test_cache_dir_is_isolated_by_hash(temp_dir, results, simple_space):
    cand_a = Candidate(simple_space, torch.tensor([5.0]))
    cand_b = Candidate(simple_space, torch.tensor([7.0]))
    h_a = compute_candidate_hash(cand_a)
    h_b = compute_candidate_hash(cand_b)
    wf_hash = "wf123"

    save_cache_entry(str(temp_dir), wf_hash, h_a, {"out": torch.tensor(1.0)})
    save_cache_entry(str(temp_dir), wf_hash, h_b, {"out": torch.tensor(2.0)})

    assert check_cache_entry(str(temp_dir), wf_hash, h_a)
    assert check_cache_entry(str(temp_dir), wf_hash, h_b)

    loaded_a = load_cache_entry(str(temp_dir), wf_hash, h_a)
    assert torch.allclose(loaded_a["out"], torch.tensor(1.0))

def test_different_workflow_hash_isolates(temp_dir, results, candidate):
    save_cache_entry(str(temp_dir), "wf_a", compute_candidate_hash(candidate), results)
    save_cache_entry(str(temp_dir), "wf_b", compute_candidate_hash(candidate), {"out": torch.tensor(99.0)})

    loaded = load_cache_entry(str(temp_dir), "wf_a", compute_candidate_hash(candidate))
    assert torch.allclose(loaded["out"], results["out"])

def test_unrelated_files_in_workdir_do_not_break_check(temp_dir, results, candidate):
    wf_hash = "stable"
    cand_hash = compute_candidate_hash(candidate)
    save_cache_entry(str(temp_dir), wf_hash, cand_hash, results)

    f = os.path.join(str(temp_dir), "unrelated.db")
    with open(f, "w") as fh:
        fh.write("some data")

    assert check_cache_entry(str(temp_dir), wf_hash, cand_hash)

    with open(f, "w") as fh:
        fh.write("more data that changes size")

    assert check_cache_entry(str(temp_dir), wf_hash, cand_hash)

def test_match_partial_all_hits(temp_dir, simple_space):
    wf = Workflow(SimpleModule())

    cand_a = Candidate(simple_space, torch.tensor([1.0]))
    cand_b = Candidate(simple_space, torch.tensor([2.0]))
    batch = CandidateBatch([cand_a, cand_b])

    cfg = CacheConfig(base_dir=str(temp_dir))
    cache = WorkflowExecutionCache(wf, config=cfg, workdir_root=str(temp_dir))
    cache.save(cand_a, {"out": torch.tensor(10.0)})
    cache.save(cand_b, {"out": torch.tensor(20.0)})

    results, misses, indices = cache.match_partial(batch)
    assert len(results) == 2
    assert len(misses) == 0
    assert len(indices) == 0
    assert results[0] is not None
    assert results[1] is not None
    assert torch.allclose(results[0]["out"], torch.tensor(10.0))
    assert torch.allclose(results[1]["out"], torch.tensor(20.0))

def test_match_partial_all_misses(temp_dir, simple_space):
    wf = Workflow(SimpleModule())

    cand_a = Candidate(simple_space, torch.tensor([1.0]))
    cand_b = Candidate(simple_space, torch.tensor([2.0]))
    cand_c = Candidate(simple_space, torch.tensor([3.0]))
    batch = CandidateBatch([cand_a, cand_b, cand_c])

    cfg = CacheConfig(base_dir=str(temp_dir))
    cache = WorkflowExecutionCache(wf, config=cfg, workdir_root=str(temp_dir))
    results, misses, indices = cache.match_partial(batch)
    assert len(results) == 3
    assert len(misses) == 3
    assert len(indices) == 3
    assert results[0] is None
    assert results[1] is None
    assert results[2] is None
    assert indices == [0, 1, 2]

def test_match_partial(temp_dir, candidate_batch, simple_space):
    wf = Workflow(SimpleModule())

    cand_a = Candidate(simple_space, torch.tensor([1.0]))
    cand_b = Candidate(simple_space, torch.tensor([2.0]))
    cand_c = Candidate(simple_space, torch.tensor([3.0]))
    batch = CandidateBatch([cand_a, cand_b, cand_c])

    cfg = CacheConfig(base_dir=str(temp_dir))
    cache = WorkflowExecutionCache(wf, config=cfg, workdir_root=str(temp_dir))
    cache.save(cand_a, {"out": torch.tensor(10.0)})
    cache.save(cand_b, {"out": torch.tensor(20.0)})

    results, misses, indices = cache.match_partial(batch)
    
    assert len(results) == 3
    assert len(misses) == 1
    assert len(indices) == 1

    assert torch.allclose(results[0]["out"], torch.tensor(10.0))
    assert torch.allclose(results[1]["out"], torch.tensor(20.0))
    assert results[2] is None
    assert misses[0].tensor.item() == 3.0
    assert indices == [2]

def test_cache_persists_across_cache_instances(temp_dir, candidate, results):
    wf = Workflow(SimpleModule())

    cfg = CacheConfig(base_dir=str(temp_dir))
    cache1 = WorkflowExecutionCache(wf, config=cfg, workdir_root=str(temp_dir))
    cache1.save(candidate, results)

    wf2 = Workflow(SimpleModule())
    cache2 = WorkflowExecutionCache(wf2, config=cfg, workdir_root=str(temp_dir))
    assert cache2.has(candidate)

def test_cache_misses_on_different_candidate(temp_dir, simple_space):
    wf = Workflow(SimpleModule())

    cand_a = Candidate(simple_space, torch.tensor([5.0]))
    cand_b = Candidate(simple_space, torch.tensor([7.0]))

    cfg = CacheConfig(base_dir=str(temp_dir))
    cache = WorkflowExecutionCache(wf, config=cfg, workdir_root=str(temp_dir))
    cache.save(cand_a, {"out": torch.tensor(1.0)})
    assert cache.has(cand_a)
    assert not cache.has(cand_b)

def test_workflow_hash_stable_with_seeding():
    seed_all(42)
    m1_a = SimpleModule()
    m2_a = AnotherModule()
    wf_a = Workflow(m1_a >> m2_a)
    rng = get_random_engine()
    rng.seed_all(wf_a)

    seed_all(42)
    m1_b = SimpleModule()
    m2_b = AnotherModule()
    wf_b = Workflow(m1_b >> m2_b)
    rng = get_random_engine()
    rng.seed_all(wf_b)

    assert compute_workflow_hash(wf_a) == compute_workflow_hash(wf_b)

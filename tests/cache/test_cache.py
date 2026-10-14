import os
import numpy as np
import torch
import pytest
import torch.autograd.forward_ad as fw

from dsbo.modules import batch_fuse
from dsbo.workflow import Workflow
from dsbo.utils.random import seed_all, get_random_engine
from dsbo.utils.cache import (
    CachePolicy,
    CacheConfig,
    module_code,
    compute_workflow_hash,
    compute_candidate_hash,
    entry_dir,
    write_npz,
    load_npz,
    save_cache_entry,
    load_cache_entry,
    check_cache_entry,
    WorkflowExecutionCache,
    RESULTS_NAME,
)
from dsbo.parameters import ParameterSpace, Vector, Continuous, Candidate, CandidateBatch
from dsbo.surrogates.graph_transformations import GraphMode

from tests.cache.conftest import (
    SimpleModule,
    AnotherModule,
    H5Producer,
    H5InputModule,
)


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


def test_cache_config_custom_base_dir():
    cfg = CacheConfig(base_dir="/custom/path")
    assert cfg.base_dir == "/custom/path"
    assert cfg.cache_policy == CachePolicy.CACHE_EVERYTHING


def test_cache_config_unknown_mode_falls_back_to_false():
    cfg = CacheConfig(cache_policy=CachePolicy.RETRAIN_SURROGATES)
    assert not cfg.should_cache(GraphMode.MODULE_SURROGATE_EVAL)


def test_module_code_metadata():
    m = SimpleModule()
    code = module_code(m)
    assert "SimpleModule" in code
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


def test_module_code_without_seed():
    code = module_code(SimpleModule(), include_seed=False)
    assert "seed=" not in code
    assert "no_grad=False" in code


def test_module_code_batch_fused():
    fused = batch_fuse(H5InputModule())
    code = module_code(fused)
    assert code.startswith("batch_fuse:")
    assert "H5InputModule" in code


def test_same_graph_same_hash():
    wf_a = Workflow(SimpleModule() >> AnotherModule())
    wf_b = Workflow(SimpleModule() >> AnotherModule())
    assert compute_workflow_hash(wf_a) == compute_workflow_hash(wf_b)


def test_different_graph_different_hash():
    wf_a = Workflow(SimpleModule() >> AnotherModule())
    wf_b = Workflow(AnotherModule() >> SimpleModule())
    assert compute_workflow_hash(wf_a) != compute_workflow_hash(wf_b)


def test_hash_is_deterministic():
    wf = Workflow(SimpleModule() >> AnotherModule())
    h1 = compute_workflow_hash(wf)
    h2 = compute_workflow_hash(wf)
    assert h1 == h2


def test_single_node_workflow_hash():
    wf = Workflow(SimpleModule())
    h = compute_workflow_hash(wf)
    assert isinstance(h, str) and len(h) > 0


def test_empty_workflow_graph_hash():
    wf = Workflow(SimpleModule())
    wf.task_graph.clear()
    h = compute_workflow_hash(wf)
    assert isinstance(h, str)


def test_batch_fuse_changes_workflow_hash():
    default = Workflow(H5InputModule())
    fused = Workflow(batch_fuse(H5InputModule()))
    assert compute_workflow_hash(default) != compute_workflow_hash(fused)


def test_batch_reduce_workflow_hash_is_deterministic():
    wf = Workflow(H5Producer() >> batch_fuse(H5InputModule()))
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


def test_candidate_hash_is_deterministic(simple_space):
    c = Candidate(simple_space, torch.tensor([5.0]))
    assert compute_candidate_hash(c) == compute_candidate_hash(c)


def test_candidate_hash_different_spaces():
    space_a = ParameterSpace(x=Vector(0, 10, 1))
    space_b = ParameterSpace(x=Continuous(0, 10))
    c_a = Candidate(space_a, torch.tensor([5.0]))
    c_b = Candidate(space_b, torch.tensor([5.0]))
    assert compute_candidate_hash(c_a) != compute_candidate_hash(c_b)


def test_candidate_hash_multi_dim_tensor():
    space = ParameterSpace(x=Vector(0, 10, 1))
    t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    c = Candidate(space, t)
    h = compute_candidate_hash(c)
    assert isinstance(h, str) and len(h) > 0
    assert compute_candidate_hash(c) == compute_candidate_hash(c)


def test_candidate_hash_includes_tangent_direction_and_magnitude():
    space = ParameterSpace(x=Vector(0, 10, 2))
    candidate = Candidate(space, torch.tensor([2.0, 4.0]))
    with fw.dual_level():
        # This set creation is used to count the number of unique hashes created.
        # It should be unique for each of the tangents listed, [1., 0.], [0., 1.], and [2., 0.]
        hashes = {
            compute_candidate_hash(candidate.set_tangent(torch.tensor(tangent)))
            for tangent in ([1.0, 0.0], [0.0, 1.0], [2.0, 0.0])
        }
    assert len(hashes) == 3


def test_candidate_hash_distinguishes_missing_and_zero_tangent(candidate):
    with fw.dual_level():
        missing_hash = compute_candidate_hash(candidate)
        zero_hash = compute_candidate_hash(candidate.set_tangent(torch.zeros_like(candidate.tensor)))
    assert missing_hash != zero_hash


def test_candidate_hash_includes_forward_ad_mode(candidate):
    primal_hash = compute_candidate_hash(candidate)
    with fw.dual_level():
        forward_hash = compute_candidate_hash(candidate)
    assert primal_hash != forward_hash
    assert compute_candidate_hash(candidate) == primal_hash


def test_candidate_hash_stable_across_dual_levels(candidate):
    hashes = []
    for _ in range(2):
        with fw.dual_level():
            dual_candidate = candidate.set_tangent(torch.ones_like(candidate.tensor))
            clone = Candidate(candidate.space, dual_candidate.tensor.clone())
            hashes.append(compute_candidate_hash(dual_candidate))
            assert compute_candidate_hash(clone) == hashes[-1]
    assert hashes[0] == hashes[1]


@pytest.mark.parametrize("tensor_a,tensor_b", [
    (torch.zeros(4), torch.zeros(2, 2)),
    (torch.zeros(2, dtype=torch.float32), torch.zeros(2, dtype=torch.int32)),
], ids=["shape", "dtype"])
def test_candidate_hash_includes_tensor_metadata(simple_space, tensor_a, tensor_b):
    assert tensor_a.numpy().tobytes() == tensor_b.numpy().tobytes()
    assert (compute_candidate_hash(Candidate(simple_space, tensor_a)) !=
            compute_candidate_hash(Candidate(simple_space, tensor_b)))


def test_entry_dir_deterministic():
    assert entry_dir("/base", "a", "b") == entry_dir("/base", "a", "b")


def test_entry_dir_differs_on_hash_change():
    assert entry_dir("/base", "a", "b") != entry_dir("/base", "a", "c")


def test_write_load_scalar_results(temp_dir, results):
    tangent_keys = write_npz(str(temp_dir), results)
    assert tangent_keys == []
    manifest = {"result_keys": ["out"], "tangent_keys": []}
    loaded = load_npz(str(temp_dir), manifest)
    assert loaded is not None
    assert torch.allclose(loaded["out"], results["out"])


def test_write_load_with_tangent(temp_dir, results_with_tangent):
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


def test_write_load_empty_results(temp_dir):
    tangent_keys = write_npz(str(temp_dir), {})
    assert tangent_keys == []
    manifest = {"result_keys": [], "tangent_keys": []}
    loaded = load_npz(str(temp_dir), manifest)
    assert loaded == {}


def test_write_load_multiple_keys(temp_dir):
    results = {"a": torch.tensor(1.0), "b": torch.tensor(2.0), "c": torch.tensor(3.0)}
    tangent_keys = write_npz(str(temp_dir), results)
    assert tangent_keys == []
    manifest = {"result_keys": ["a", "b", "c"], "tangent_keys": []}
    loaded = load_npz(str(temp_dir), manifest)
    assert loaded is not None
    assert torch.allclose(loaded["a"], torch.tensor(1.0))
    assert torch.allclose(loaded["b"], torch.tensor(2.0))
    assert torch.allclose(loaded["c"], torch.tensor(3.0))


@pytest.mark.parametrize("value", [42, 42.0, True, [1, 2, 3], np.array([1, 2, 3]), None],
                         ids=["int", "float", "bool", "list", "numpy-array", "none"])
def test_write_rejects_non_tensor_results(temp_dir, value):
    results = {"tensor": torch.tensor(1.0), "invalid": value}

    with pytest.raises(TypeError, match=f"'invalid' must be a torch\.Tensor, got {type(value).__name__}"):
        write_npz(str(temp_dir), results)

    # Ensure no file was written
    assert not os.path.isfile(os.path.join(str(temp_dir), RESULTS_NAME))


def test_save_load_tensor_subclass(temp_dir):
    result = torch.nn.Parameter(torch.tensor(7.0))
    save_cache_entry(str(temp_dir), "wf1", "cand1", {"out": result})
    loaded = load_cache_entry(str(temp_dir), "wf1", "cand1")
    assert torch.allclose(loaded["out"], result)


def test_write_load_tangent_and_primal_mixed(temp_dir):
    with fw.dual_level():
        results = {
            "a": fw.make_dual(torch.tensor(1.0), torch.tensor(10.0)),
            "b": torch.tensor(2.0),
        }
        tangent_keys = write_npz(str(temp_dir), results)
    assert "a" in tangent_keys
    assert "b" not in tangent_keys
    manifest = {"result_keys": ["a", "b"], "tangent_keys": tangent_keys}
    with fw.dual_level():
        loaded = load_npz(str(temp_dir), manifest)
        assert loaded is not None
        a_primal, a_tangent = fw.unpack_dual(loaded["a"])
        assert torch.allclose(a_primal, torch.tensor(1.0))
        assert torch.allclose(a_tangent, torch.tensor(10.0))
        assert torch.allclose(loaded["b"], torch.tensor(2.0))


def test_load_tangent_without_dual_level_returns_miss(temp_dir):
    with fw.dual_level():
        save_cache_entry(str(temp_dir), "wf1", "cand1", {
            "out": fw.make_dual(torch.tensor(10.0), torch.tensor(2.0)),
            "constant": torch.tensor(7.0),
        })
    assert check_cache_entry(str(temp_dir), "wf1", "cand1")
    manifest = {"result_keys": ["out", "constant"], "tangent_keys": ["out"]}
    path = entry_dir(str(temp_dir), "wf1", "cand1")
    assert load_npz(path, manifest) is None
    assert load_cache_entry(str(temp_dir), "wf1", "cand1") is None


def test_check_missing_entry(temp_dir):
    assert not check_cache_entry(str(temp_dir), "nosuch", "nosuch")


def test_load_missing_entry(temp_dir):
    assert load_cache_entry(str(temp_dir), "nosuch", "nosuch") is None


def test_save_load_round_trip(temp_dir):
    save_cache_entry(str(temp_dir), "wf1", "cand1", {"out": torch.tensor(7.0)})
    assert check_cache_entry(str(temp_dir), "wf1", "cand1")
    loaded = load_cache_entry(str(temp_dir), "wf1", "cand1")
    assert torch.allclose(loaded["out"], torch.tensor(7.0))


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


def test_cache_default_config_uses_workdir_root(temp_dir):
    cache = WorkflowExecutionCache(Workflow(SimpleModule()), workdir_root=str(temp_dir))
    assert cache.config.base_dir == str(temp_dir)


def test_cache_config_overrides_workdir_root(temp_dir):
    cfg = CacheConfig(base_dir="/custom")
    cache = WorkflowExecutionCache(Workflow(SimpleModule()), config=cfg, workdir_root=str(temp_dir))
    assert cache._effective_base == "/custom"


def test_has_all_true(temp_dir, simple_space):
    cfg = CacheConfig(base_dir=str(temp_dir))
    cache = WorkflowExecutionCache(Workflow(SimpleModule()), config=cfg, workdir_root=str(temp_dir))
    c1 = Candidate(simple_space, torch.tensor([1.0]))
    c2 = Candidate(simple_space, torch.tensor([2.0]))
    cache.save(c1, {"out": torch.tensor(10.0)})
    cache.save(c2, {"out": torch.tensor(20.0)})
    assert cache.has_all([c1, c2])


def test_has_all_false(temp_dir, simple_space):
    cfg = CacheConfig(base_dir=str(temp_dir))
    cache = WorkflowExecutionCache(Workflow(SimpleModule()), config=cfg, workdir_root=str(temp_dir))
    c1 = Candidate(simple_space, torch.tensor([1.0]))
    c2 = Candidate(simple_space, torch.tensor([2.0]))
    cache.save(c1, {"out": torch.tensor(10.0)})
    assert not cache.has_all([c1, c2])


def test_load_all_success(temp_dir, simple_space):
    cfg = CacheConfig(base_dir=str(temp_dir))
    cache = WorkflowExecutionCache(Workflow(SimpleModule()), config=cfg, workdir_root=str(temp_dir))
    c1 = Candidate(simple_space, torch.tensor([1.0]))
    c2 = Candidate(simple_space, torch.tensor([2.0]))
    cache.save(c1, {"out": torch.tensor(10.0)})
    cache.save(c2, {"out": torch.tensor(20.0)})
    results = cache.load_all([c1, c2])
    assert torch.allclose(results[0]["out"], torch.tensor(10.0))
    assert torch.allclose(results[1]["out"], torch.tensor(20.0))


def test_load_all_raises_on_missing(temp_dir, simple_space):
    cfg = CacheConfig(base_dir=str(temp_dir))
    cache = WorkflowExecutionCache(Workflow(SimpleModule()), config=cfg, workdir_root=str(temp_dir))
    c1 = Candidate(simple_space, torch.tensor([1.0]))
    c2 = Candidate(simple_space, torch.tensor([2.0]))
    cache.save(c1, {"out": torch.tensor(10.0)})
    with pytest.raises(Exception, match="No cached result"):
        cache.load_all([c1, c2])


def test_save_all(temp_dir, simple_space):
    cfg = CacheConfig(base_dir=str(temp_dir))
    cache = WorkflowExecutionCache(Workflow(SimpleModule()), config=cfg, workdir_root=str(temp_dir))
    c1 = Candidate(simple_space, torch.tensor([1.0]))
    c2 = Candidate(simple_space, torch.tensor([2.0]))
    cache.save_all([c1, c2], [{"out": torch.tensor(10.0)}, {"out": torch.tensor(20.0)}])
    assert cache.has(c1)
    assert cache.has(c2)


def test_match_partial_all_hits(temp_dir, simple_space):
    cfg = CacheConfig(base_dir=str(temp_dir))
    cache = WorkflowExecutionCache(Workflow(SimpleModule()), config=cfg, workdir_root=str(temp_dir))

    cand_a = Candidate(simple_space, torch.tensor([1.0]))
    cand_b = Candidate(simple_space, torch.tensor([2.0]))
    batch = CandidateBatch([cand_a, cand_b])

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
    cfg = CacheConfig(base_dir=str(temp_dir))
    cache = WorkflowExecutionCache(Workflow(SimpleModule()), config=cfg, workdir_root=str(temp_dir))

    cand_a = Candidate(simple_space, torch.tensor([1.0]))
    cand_b = Candidate(simple_space, torch.tensor([2.0]))
    cand_c = Candidate(simple_space, torch.tensor([3.0]))
    batch = CandidateBatch([cand_a, cand_b, cand_c])

    results, misses, indices = cache.match_partial(batch)
    assert len(results) == 3
    assert len(misses) == 3
    assert len(indices) == 3
    assert results[0] is None
    assert results[1] is None
    assert results[2] is None
    assert indices == [0, 1, 2]


def test_match_partial(temp_dir, candidate_batch, simple_space):
    cfg = CacheConfig(base_dir=str(temp_dir))
    cache = WorkflowExecutionCache(Workflow(SimpleModule()), config=cfg, workdir_root=str(temp_dir))

    cand_a = Candidate(simple_space, torch.tensor([1.0]))
    cand_b = Candidate(simple_space, torch.tensor([2.0]))
    cand_c = Candidate(simple_space, torch.tensor([3.0]))
    batch = CandidateBatch([cand_a, cand_b, cand_c])

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


def test_match_partial_empty_result_is_hit(temp_dir, candidate):
    cache = WorkflowExecutionCache(Workflow(SimpleModule()), workdir_root=str(temp_dir))
    cache.save(candidate, {})
    results, misses, indices = cache.match_partial([candidate])
    assert results == [{}]
    assert misses == []
    assert indices == []


def test_load_or_compute_recomputes_rejected_entries(temp_dir, simple_space):
    cache = WorkflowExecutionCache(Workflow(SimpleModule()), workdir_root=str(temp_dir))
    candidates = [
        Candidate(simple_space, torch.tensor([value])) for value in (1.0, 2.0, 3.0)
    ]
    cache.save(candidates[1], {"out": torch.tensor(20.0)})

    # Put tangent-containing results under a primal key to exercise loader rejection.
    primal_hash = compute_candidate_hash(candidates[0])
    with fw.dual_level():
        save_cache_entry(str(temp_dir), cache.workflow_hash, primal_hash, {
            "out": fw.make_dual(torch.tensor(10.0), torch.tensor(1.0)),
        })
    # The key exists...
    assert cache.has(candidates[0])
    # ... but only with a dual, so outside of the dual_level() context, the result is still None
    assert cache.load(candidates[0]) is None

    results, misses, indices = cache.match_partial(candidates)
    assert results[0] is None
    assert torch.equal(results[1]["out"], torch.tensor(20.0))
    assert results[2] is None
    assert misses == [candidates[0], candidates[2]]
    assert indices == [0, 2]

    computed_candidates = []

    def compute(misses):
        computed_candidates.extend(misses)
        return [{"out": candidate.tensor[0] * 10} for candidate in misses]

    results = cache.load_or_compute(candidates, compute)
    assert computed_candidates == [candidates[0], candidates[2]]
    assert [result["out"].item() for result in results] == [10.0, 20.0, 30.0]
    assert torch.equal(cache.load(candidates[0])["out"], torch.tensor(10.0))
    assert cache.has_all(candidates)


def test_directional_cache_round_trip_across_instances_and_dual_levels(temp_dir):
    space = ParameterSpace(x=Vector(0, 10, 2))
    candidate = Candidate(space, torch.tensor([2.0, 4.0]))
    cache = WorkflowExecutionCache(Workflow(SimpleModule()), workdir_root=str(temp_dir))
    directions = [(torch.tensor([1.0, 0.0]), 4.0), (torch.tensor([0.0, 1.0]), 3.0)]

    with fw.dual_level():
        for tangent, _ in directions:
            dual_candidate = candidate.set_tangent(tangent)
            assert not cache.has(dual_candidate)
            x, y = dual_candidate.tensor.unbind()
            cache.save(dual_candidate, {"out": x ** 2 + 3 * y, "constant": torch.tensor(7.0)})

    cache = WorkflowExecutionCache(Workflow(SimpleModule()), workdir_root=str(temp_dir))
    with fw.dual_level():
        for tangent, expected in directions:
            dual_candidate = candidate.set_tangent(tangent)
            assert cache.has(dual_candidate)
            loaded = cache.load(dual_candidate)
            assert loaded is not None
            primal, loaded_tangent = fw.unpack_dual(loaded["out"])
            assert torch.equal(primal, torch.tensor(16.0))
            assert torch.equal(loaded_tangent, torch.tensor(expected))
            assert fw.unpack_dual(loaded["constant"]).tangent is None


def test_load_or_compute_merges_and_caches_misses(temp_dir, simple_space):
    cfg = CacheConfig(base_dir=str(temp_dir))
    cache = WorkflowExecutionCache(Workflow(SimpleModule()), config=cfg, workdir_root=str(temp_dir))
    candidates = [
        Candidate(simple_space, torch.tensor([1.0])),
        Candidate(simple_space, torch.tensor([2.0])),
        Candidate(simple_space, torch.tensor([3.0])),
    ]
    cache.save(candidates[1], {"out": torch.tensor(20.0)})

    computed_candidates = []

    def compute(misses):
        computed_candidates.extend(misses)
        return [{"out": candidate.tensor[0] * 10} for candidate in misses]

    results = cache.load_or_compute(candidates, compute)

    assert computed_candidates == [candidates[0], candidates[2]]
    assert [result["out"].item() for result in results] == [10.0, 20.0, 30.0]
    assert cache.has(candidates[0])
    assert cache.has(candidates[2])


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


def test_cache_load_returns_none_on_miss(temp_dir, simple_space):
    cfg = CacheConfig(base_dir=str(temp_dir))
    cache = WorkflowExecutionCache(Workflow(SimpleModule()), config=cfg, workdir_root=str(temp_dir))
    c = Candidate(simple_space, torch.tensor([5.0]))
    assert cache.load(c) is None


def test_cache_save_overwrites_existing(temp_dir, simple_space):
    cfg = CacheConfig(base_dir=str(temp_dir))
    cache = WorkflowExecutionCache(Workflow(SimpleModule()), config=cfg, workdir_root=str(temp_dir))
    c = Candidate(simple_space, torch.tensor([5.0]))
    cache.save(c, {"out": torch.tensor(1.0)})
    cache.save(c, {"out": torch.tensor(99.0)})
    loaded = cache.load(c)
    assert torch.allclose(loaded["out"], torch.tensor(99.0))


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

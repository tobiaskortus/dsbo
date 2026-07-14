from __future__ import annotations

import os
import json
import torch
import hashlib
import inspect
import logging
import marshal
import numpy as np
from enum import Enum
import networkx as nx
from dataclasses import dataclass
from typing import Any, Sequence, TYPE_CHECKING
import torch.autograd.forward_ad as fw

from dsbo.modules import Module
from dsbo.surrogates.graph_transformations import GraphMode
from dsbo.parameters import Candidate, CandidateBatch

if TYPE_CHECKING:
    from dsbo.workflow import Workflow, ResultsBatch


CACHE_DIR = ".dsbo_cache"
MANIFEST_NAME = "manifest.json"
RESULTS_NAME = "results.npz"

class CachePolicy(Enum):
    ALWAYS_EXECUTE = "always_execute"
    CACHE_EVERYTHING = "cache_everything"
    RETRAIN_SURROGATES = "retrain_surrogates"


# TODO: implement cleanup policy
# class CacheCleanupPolicy(Enum):
#     KEEP_ALL = "keep_all"
#     KEEP_MANIFEST = "keep_manifest"

@dataclass
class CacheConfig:
    base_dir: str = ".dsbo_cache"
    cache_policy: CachePolicy = CachePolicy.CACHE_EVERYTHING
    # cleanup_policy: CacheCleanupPolicy = CacheCleanupPolicy.KEEP_ALL

    def should_cache(self, mode: GraphMode = GraphMode.DEFAULT) -> bool:
        if self.cache_policy == CachePolicy.CACHE_EVERYTHING:
            return True
        if self.cache_policy == CachePolicy.RETRAIN_SURROGATES:
            result = (mode == GraphMode.DEFAULT) | (mode == GraphMode.MODULE_SURROGATE_TRAIN)
            return result
        return False
    

def code_hash(code_obj):
    #TODO: replace this simplified heuristic with an improved code checking
    #to ensure that the output is the same when using the cache.
    h = hashlib.sha256()
    h.update(code_obj.co_code)
    h.update(str(code_obj.co_argcount).encode())
    h.update(str(code_obj.co_nlocals).encode())
    h.update(str(code_obj.co_stacksize).encode())
    h.update(str(code_obj.co_flags).encode())
    h.update(marshal.dumps(code_obj.co_names))
    h.update(marshal.dumps(code_obj.co_varnames))
    h.update(marshal.dumps(code_obj.co_consts))
    h.update(marshal.dumps(code_obj.co_cellvars))
    h.update(marshal.dumps(code_obj.co_freevars))
    return h.digest()  # Return bytes, not hex string


def module_code(node: Module, include_seed: bool = True) -> str:
    cls = type(node)
    identity = f"{cls.__module__}.{cls.__qualname__}"
    state = f"seed={node._seed},no_grad={node._no_grad_op}" if include_seed else f"no_grad={node._no_grad_op}"
    code_digest = hashlib.sha256()
    for _, m in inspect.getmembers(cls, predicate=inspect.isfunction):
        code_digest.update(code_hash(m.__code__))
    return f"{identity}:{code_digest.hexdigest()}:{state}"


def compute_workflow_hash(workflow: Workflow) -> str:
    G = workflow.task_graph.copy()
    for node in G.nodes():
        if isinstance(node, Module):
            G.nodes[node]["_dsbo_code"] = module_code(node, include_seed=True)
        else:
            G.nodes[node]["_dsbo_code"] = str(type(node).__name__)
    return nx.weisfeiler_lehman_graph_hash(G, node_attr="_dsbo_code")


def compute_candidate_hash(candidate: Candidate) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(candidate.space._spec, sort_keys=True, default=vars).encode())
    h.update(candidate.tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def entry_dir(base_dir: str, wf_hash: str, cand_hash: str) -> str:
    combined_hash = hashlib.sha256(wf_hash.encode() + cand_hash.encode()).hexdigest()
    return os.path.join(base_dir, CACHE_DIR, combined_hash)


def write_npz(cache_entry_path: str, results: dict[str, Any]) -> list[str]:
    npz: dict[str, np.ndarray] = {}
    tangent_keys: list[str] = []
    for k, v in results.items():
        if isinstance(v, torch.Tensor):
            dual = fw.unpack_dual(v)
            npz[f"{k}_primal"] = dual.primal.detach().cpu().numpy()
            if dual.tangent is not None:
                npz[f"{k}_tangent"] = dual.tangent.detach().cpu().numpy()
                tangent_keys.append(k)
        else:
            npz[k] = np.asarray(v)
    np.savez(os.path.join(cache_entry_path, RESULTS_NAME), **npz)
    return tangent_keys


def load_npz(cache_entry_path: str, manifest: dict[str, Any]) -> dict[str, Any] | None:
    rp = os.path.join(cache_entry_path, RESULTS_NAME)
    if not os.path.isfile(rp):
        return None

    data = np.load(rp)
    tangent_set = set(manifest.get("tangent_keys", []))
    results: dict[str, Any] = {}
    for k in manifest["result_keys"]:
        primal = torch.from_numpy(data[f"{k}_primal"])
        if k in tangent_set:
            tangent = torch.from_numpy(data[f"{k}_tangent"])
            results[k] = fw.make_dual(primal, tangent)
        else:
            results[k] = primal
    return results


def save_cache_entry(base_dir: str, workflow_hash: str, candidate_hash: str,
                      results: dict[str, Any]) -> None:
    cache_entry_path = entry_dir(base_dir, workflow_hash, candidate_hash)
    os.makedirs(cache_entry_path, exist_ok=True)

    tangent_keys = write_npz(cache_entry_path, results)

    manifest = {
        "workflow_hash": workflow_hash,
        "candidate_hash": candidate_hash,
        "result_keys": list(results.keys()),
        "tangent_keys": tangent_keys,
        "workdir_root": base_dir,
    }

    with open(os.path.join(cache_entry_path, MANIFEST_NAME), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)


def load_cache_entry(base_dir: str, workflow_hash: str, candidate_hash: str) -> dict[str, Any] | None:

    cache_entry_path = entry_dir(base_dir, workflow_hash, candidate_hash)
    manifest_fp = os.path.join(cache_entry_path, MANIFEST_NAME)
    if not os.path.isfile(manifest_fp):
        return None

    with open(manifest_fp) as f:
        manifest = json.load(f)

    result = load_npz(cache_entry_path, manifest)
    return result


def check_cache_entry(base_dir: str, workflow_hash: str, candidate_hash: str) -> bool:
    cache_entry_path = entry_dir(base_dir, workflow_hash, candidate_hash)
    manifest_fp = os.path.join(cache_entry_path, MANIFEST_NAME)
    results_fp = os.path.join(cache_entry_path, RESULTS_NAME)
    hit = os.path.isfile(manifest_fp) and os.path.isfile(results_fp)
    return hit


Candidates = Sequence[Candidate] | CandidateBatch


class TaskExecutionCache:
    # TODO: This probably requires a custom scheduler implementation to support 
    # caching when workers with results are lost and require re-computing.
    def __init__(self, workflow: Workflow, config: CacheConfig | None = None, workdir_root: str | None = None):
        self.workflow_hash = compute_workflow_hash(workflow)

    def has(self, task_id: str) -> bool:
        return False
    
    def load(self, task_id: str) -> bool:
        return {}
    
    def save(self, task_id, results: dict[str, Any]) -> None:
        return {}



class WorkflowExecutionCache:
    def __init__(self, workflow: Workflow, config: CacheConfig | None = None, workdir_root: str | None = None,):
        self._effective_base = (
            config.base_dir
            if config is not None and config.base_dir is not None
            else workdir_root
        )
        self.config = config or CacheConfig(base_dir=self._effective_base)
        self.workflow_hash = compute_workflow_hash(workflow)

    def has(self, candidate: Candidate) -> bool:
        return check_cache_entry(
            self._effective_base,
            self.workflow_hash,
            compute_candidate_hash(candidate),
        )

    def load(self, candidate: Candidate) -> dict[str, Any] | None:
        return load_cache_entry(
            self._effective_base,
            self.workflow_hash,
            compute_candidate_hash(candidate),
        )

    def save(self, candidate: Candidate, results: dict[str, Any]) -> None:
        save_cache_entry(
            self._effective_base,
            self.workflow_hash,
            compute_candidate_hash(candidate),
            results,
        )

    def has_all(self, candidates: Candidates) -> bool:
        return all(self.has(c) for c in candidates)
    
    def match_partial(self, candidates: Candidates) -> tuple[list[dict | None], list[Candidate] , list[int]]:
        results = [None] * len(candidates)
        misses  = []
        indices = []

        for i, c in enumerate(candidates):
            if self.has(c):
                results[i] =  self.load(c)
                continue
            
            indices.append(i)
            misses.append(c)

        return results, misses, indices

    def load_all(self, candidates: Candidates) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for c in candidates:
            r = self.load(c)
            if r is None:
                raise Exception(
                    f"No cached result for candidate hash "
                    f"{compute_candidate_hash(c)[:12]}..."
                )
            results.append(r)

        return results

    def save_all(self, candidates: Candidates, batch_results: list[dict[str, Any]]):
        for c, r in zip(candidates, batch_results):
            self.save(c, r)

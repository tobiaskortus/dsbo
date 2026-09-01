from __future__ import annotations

import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    DistributedSampler,
    IterableDataset,
    RandomSampler,
    SequentialSampler,
)

from dsbo.surrogates.easyscale.random import (
    VirtualRankContext,
    initial_context,
    seed_everything,
    seed_scope,
)


@dataclass(frozen=True)
class VirtualBatch:
    ranks: tuple[int, ...]
    indices: tuple[tuple[int, ...], ...]
    batches: tuple[Any, ...]

    def items(self):
        return zip(self.ranks, self.batches)


class VirtualDataset(Dataset):
    def __init__(self, dataset: Dataset, seed: int):
        self.dataset = dataset
        self.seed = seed

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, key: tuple[int, int, int, int, int]):
        rank, epoch, batch_position, sample_offset, index = key
        seed_sequence = np.random.SeedSequence([self.seed, *key])
        sample_seed = int(seed_sequence.generate_state(1, dtype=np.uint32)[0])

        with seed_scope(sample_seed, torch.device("cpu")):
            sample = self.dataset[index]

        return rank, index, sample


class SeedWorker:
    def __init__(self, worker_init_fn: Callable[[int], None] | None):
        self.worker_init_fn = worker_init_fn

    def __call__(self, worker_id: int) -> None:
        worker_seed = torch.initial_seed() % 2**32
        seed_everything(worker_seed)
        if self.worker_init_fn is not None:
            self.worker_init_fn(worker_id)


class CollateTagged:
    def __init__(self, ranks: tuple[int, ...], collate_fn: Callable):
        self.ranks = ranks
        self.collate_fn = collate_fn

    def __call__(self, samples) -> VirtualBatch:
        indices: dict[int, list[int]] = {rank: [] for rank in self.ranks}
        grouped: dict[int, list[Any]] = {rank: [] for rank in self.ranks}
        for rank, index, sample in samples:
            indices[rank].append(index)
            grouped[rank].append(sample)
        return VirtualBatch(
            ranks=self.ranks,
            indices=tuple(tuple(indices[rank]) for rank in self.ranks),
            batches=tuple(self.collate_fn(grouped[rank]) for rank in self.ranks),
        )


class VirtualBatchSampler:
    def __init__(
        self,
        dataset: Dataset,
        virtual_world_size: int,
        assigned_ranks: Sequence[int],
        batch_size: int,
        seed: int,
        drop_last: bool,
        shuffle: bool,
        contexts: Mapping[int, VirtualRankContext],
    ) -> None:
        self.seed = seed
        self.shuffle = shuffle
        self.dataset = dataset
        self.drop_last = drop_last
        self.batch_size = batch_size
        self.contexts = dict(contexts)
        self.virtual_world_size = virtual_world_size
        self.assigned_ranks = tuple(sorted(assigned_ranks))
        self.distributed_samplers = {
            rank: DistributedSampler(
                self.dataset,
                num_replicas=self.virtual_world_size,
                rank=rank,
                shuffle=self.shuffle,
                seed=self.seed,
                drop_last=self.drop_last,
            )
            for rank in self.assigned_ranks
        }

    def _rank_batches(self, rank: int) -> BatchSampler:
        context = self.contexts[rank]
        sampler = self.distributed_samplers[rank]
        sampler.set_epoch(context.sampler_epoch)
        return BatchSampler(sampler, self.batch_size, self.drop_last)

    def __iter__(self):
        positions = {self.contexts[rank].sampler_position for rank in self.assigned_ranks}
        epochs = {self.contexts[rank].sampler_epoch for rank in self.assigned_ranks}

        if len(positions) != 1 or len(epochs) != 1:
            raise RuntimeError("assigned EST sampler contexts are not at one boundary.")
        
        iterators = []
        epoch = epochs.pop()
        position = positions.pop()
        total = len(self._rank_batches(self.assigned_ranks[0]))

        for rank in self.assigned_ranks:
            iterator = iter(self._rank_batches(rank))
            for _ in range(position):
                try:
                    next(iterator)
                except StopIteration as exc:
                    raise RuntimeError("EST sampler position exceeds the epoch.") from exc
                
            iterators.append(iterator)
            
        
        for batch_position, rank_batches in enumerate(zip(*iterators), start=position):
            for rank in self.assigned_ranks:
                self.contexts[rank].sampler_position += 1
                if self.contexts[rank].sampler_position == total:
                    self.contexts[rank].sampler_epoch = epoch + 1
                    self.contexts[rank].sampler_position = 0
            yield [
                (rank, epoch, batch_position, sample_offset, index)
                for rank, indices in zip(self.assigned_ranks, rank_batches)
                for sample_offset, index in enumerate(indices)
            ]

    def __len__(self) -> int:
        if not self.assigned_ranks:
            return 0
        total = len(self._rank_batches(self.assigned_ranks[0]))
        return max(0, total - self.contexts[self.assigned_ranks[0]].sampler_position)


def build_virtual_dataloader(
    loader: DataLoader,
    virtual_world_size: int,
    seed: int,
    assigned_ranks: Sequence[int] | None = None,
    contexts: Sequence[VirtualRankContext] | Mapping[int, VirtualRankContext] | None = None,
) -> DataLoader:
    
    assigned = tuple(range(virtual_world_size) if assigned_ranks is None else assigned_ranks)
    if contexts is None:
        context_by_rank = {
            rank: initial_context(rank, seed) for rank in range(virtual_world_size)
        }
    elif isinstance(contexts, Mapping):
        context_by_rank = dict(contexts)
    else:
        context_by_rank = {context.rank: context for context in contexts}

    batch_sampler = VirtualBatchSampler(
        dataset=loader.dataset,
        virtual_world_size=virtual_world_size,
        assigned_ranks=assigned,
        batch_size=loader.batch_size,
        seed=seed,
        drop_last=loader.drop_last,
        shuffle=isinstance(loader.sampler, RandomSampler),
        contexts={rank: context_by_rank[rank] for rank in assigned},
    )

    loader_options = {
        "batch_sampler": batch_sampler,
        "collate_fn": CollateTagged(assigned, loader.collate_fn),
        "num_workers": loader.num_workers,
        "worker_init_fn": SeedWorker(loader.worker_init_fn),
        "generator": torch.Generator().manual_seed(seed),
        "persistent_workers": loader.persistent_workers,
        "timeout": loader.timeout,
    }

    if loader.num_workers:
        loader_options["prefetch_factor"] = loader.prefetch_factor
        loader_options["multiprocessing_context"] = loader.multiprocessing_context
    
    return DataLoader(
        VirtualDataset(loader.dataset, seed),
        **loader_options,
    )

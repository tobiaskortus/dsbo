import os
import asyncio
import logging
from contextlib import nullcontext
from typing import List, Any, Tuple, Dict, Optional, Iterable

from dsbo.logger import WorkflowTaskLogger
from dsbo.utils.io import Workdir
from dsbo.workflow import Workflow
from dsbo.worker.dask import WorkerContext



async def indexed_coroutine(idx, workflow: Workflow, design: dict, workdir: str, 
                            keep_workdir: bool = False) -> Tuple[int, Any, Optional[Exception]]:
    try:
        workdir = os.path.join(workdir, f"design_{idx}")
        with Workdir.create(workdir, keep=keep_workdir):
            result = await asyncio.to_thread(workflow.run, design)
            return idx, result, None
    except Exception as e:
        return idx, None, e


class WorkflowExecutor:
    def __init__(self, worker = None):
        self.worker = worker

    def optional_worker_context(self):
        if self.worker is None:
            return nullcontext()
        
        return WorkerContext.use_worker(self.worker)

    def execute_workflow(self):
        raise NotImplementedError()

    async def execute_workflow_async(self):
        raise NotImplementedError()

    async def execute_workflow_batch_async(self, workflow: Workflow, designs: Iterable[dict],
                                           workdir: str, keep_workdir: bool = False) -> List[Tuple[int, Any]]:
        with self.optional_worker_context():
            indexed_tasks = [asyncio.create_task(indexed_coroutine(idx, workflow, design, workdir, keep_workdir)) \
                             for idx, design in enumerate(designs)]

            results = [None] * len(indexed_tasks)
            catch = (RuntimeError, ValueError)

            logger = WorkflowTaskLogger.get_current()

            for completed in asyncio.as_completed(indexed_tasks):
                idx, result, exc = await completed
                if exc is not None:
                    if isinstance(exc, catch):
                        logger.log(f"Caught exception at task idx {idx}: {exc}", logging.ERROR)
                        results[idx] = None
                    else:
                        raise exc
                else:
                    results[idx] = (idx, result)

            return list(filter(None, results))

    def execute_workflow_batch(self, workflow: Workflow, designs: Iterable[dict], workdir: str, keep_workdir: bool = False) -> List[Tuple[int, Any]]:
        try:
            return asyncio.run(self.execute_workflow_batch_async(workflow, designs, workdir, keep_workdir))
        except RuntimeError as e:
            if e.args and "asyncio.run() cannot be called from a running event loop" in e.args[0]:
                print("Cannot run synchronous execute_workflow_batch from within a jupyter notebook . Use execute_workflow_async instead.")
                return []
            else:
                raise e

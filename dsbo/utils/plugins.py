import time
import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Dict, Optional
from distributed.diagnostics.plugin import SchedulerPlugin

from dsbo.storage.models import JobStatus
from dsbo.storage import TransitionWriter


class ProgressBarPlugin(SchedulerPlugin):
    def __init__(self, progress_callback: Callable[[], None]) -> None:
        super().__init__()
        self.progress_callback = progress_callback

    def transition(self, key, start, finish, *args, **kwargs):
        match (start, finish):
            case (_, "memory") | (_, "erred"):
                self.progress_callback()


class TaskStreamPlugin(SchedulerPlugin):
    def __init__(
        self,
        writer: TransitionWriter,
        task_map: Dict[str, str | tuple[str, ...]],
        name_map: Dict[str, str],
        tag: str = "",
    ) -> None:
        super().__init__()
        self._tag = tag
        self._writer = writer
        self._task_map: Dict[str, str | tuple[str, ...]] = task_map
        self._name_map: Dict[str, str] = name_map
        self._log_buffer: Dict[str, list] = {}
        self._scheduler = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dsbo-task-stream")

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("_executor", None)
        state.pop("scheduler", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="dsbo-task-stream")
        self._scheduler = None

    def start(self, scheduler):
        self._scheduler = scheduler

    async def close(self):
        await asyncio.to_thread(self._executor.shutdown, wait=True)

    def _report_write_error(future: Future) -> None:
        error = future.exception()
        if error is not None:
            print(f"[TaskStreamPlugin] write error: {error}")

    def _write_async(self, *args) -> None:
        self._executor.submit(self._writer, *args).add_done_callback(self._report_write_error)

    def update(self, writer, task_map, name_map, tag: str = "") -> None:
        self._tag = tag
        self._writer = writer
        self._task_map = task_map
        self._name_map = name_map
        self._log_buffer.clear()

    def log_event(self, topic: str, msg):
        if topic == "runtimes" and isinstance(msg, dict) and msg.get("task_log"):
            self._log_buffer.setdefault(msg.get("task"), []).append(msg["task_log"])

    def transition(self, key, start, finish, *args, **kwargs):
        if key not in self._task_map:
            return

        def trigger_on_transition(status: JobStatus) -> None:
            try:
                if status == JobStatus.RUNNING or status == JobStatus.QUEUED:
                    ts = self._scheduler.tasks[key]
                    worker = None
                    if ts is not None and ts.processing_on is not None:
                        worker = ts.processing_on

                    name = self._name_map.get(key, key)
                    approx_start = time.time() if status == JobStatus.RUNNING else -1
                    exec_ids = self._task_map[key]
                    if isinstance(exec_ids, str):
                        exec_ids = (exec_ids,)
                    for exec_id in exec_ids:
                        self._write_async(key, name, worker.name if worker != None else "unknown", approx_start, -1, exec_id, "", status, self._tag)
                    return

                worker = self._scheduler.workers[kwargs["worker"]]
                startstop = kwargs["startstops"][0]
                name = self._name_map.get(key, key)
                messages = self._log_buffer.pop(key, [])
                log = "\n".join(messages)

                exec_ids = self._task_map[key]
                if isinstance(exec_ids, str):
                    exec_ids = (exec_ids,)
                for exec_id in exec_ids:
                    self._write_async(key, name, worker.name, startstop["start"], startstop["stop"], exec_id, log, status, self._tag)
            except Exception as e:
                print(f"[TaskStreamPlugin] transition error: {e}")

        match (start, finish):
            case (_, "processing"):
                trigger_on_transition(status=JobStatus.RUNNING)
            case (_, "queued"):
                trigger_on_transition(status=JobStatus.QUEUED)
            case (_, "memory"):
                trigger_on_transition(status=JobStatus.SUCCESS)
            case (_, "erred"):
                trigger_on_transition(status=JobStatus.FAILED)
            case (_, _):
                pass

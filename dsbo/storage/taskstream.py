from typing import Dict, Optional
from distributed.diagnostics.plugin import SchedulerPlugin

from dsbo.storage.models import JobStatus
from dsbo.storage import TransitionWriter

class TaskStreamPlugin(SchedulerPlugin):
    def __init__(
        self,
        writer: TransitionWriter,
        task_map: Dict[str, str],
        name_map: Dict[str, str],
    ) -> None:
        super().__init__()
        self.writer = writer
        self.task_map: Dict[str, str] = task_map
        self.name_map: Dict[str, str] = name_map
        self.log_buffer: Dict[str, list] = {}

    def start(self, scheduler):
        self.scheduler = scheduler

    def log_event(self, topic: str, msg):
        if topic == "runtimes" and isinstance(msg, dict) and msg.get("task_log"):
            self.log_buffer.setdefault(msg.get("task"), []).append(msg["task_log"])

    def transition(self, key, start, finish, *args, **kwargs):

        def trigger_on_transition(status: JobStatus) -> None:
            try:
                worker = self.scheduler.workers[kwargs["worker"]]
                startstop = kwargs["startstops"][0]
                exec_id = self.task_map[key]
                name    = self.name_map[key]
                messages = self.log_buffer.pop(key, [])
                log = "\n".join(messages)

                self.writer(key, name, worker.name, startstop["start"], startstop["stop"], exec_id, log, status)
            except Exception as e:
                print(f"[TaskStreamPlugin] transition error: {e}")


        match (start, finish):
            case (_, "memory"):
                trigger_on_transition(status=JobStatus.SUCCESS)
            case (_, "erred"): 
                trigger_on_transition(status=JobStatus.FAILED)

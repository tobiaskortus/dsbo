from typing import Callable, Dict, Optional
from distributed.diagnostics.plugin import SchedulerPlugin


class TaskStreamPlugin(SchedulerPlugin):
    def __init__(
        self,
        on_transition: Callable[[str, str, float, float, Optional[str]], None],
        task_map: Dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.on_transition = on_transition
        self.task_map: Dict[str, str] = task_map if task_map is not None else {}

    def start(self, scheduler):
        self.scheduler = scheduler

    def transition(self, key, start, finish, *args, **kwargs):
        if finish != "memory":
            return
        try:
            ws = self.scheduler.workers[kwargs["worker"]]
            startstop = kwargs["startstops"][0]
            exec_id = self.task_map.get(key)
            self.on_transition(key, ws.name, startstop["start"], startstop["stop"], exec_id)
        except Exception as e:
            print(f"[TaskStreamPlugin] transition error: {e}")

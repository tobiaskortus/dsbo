import dask
import logging
from time import time
from typing import Optional
from dask.distributed import get_worker
from dsbo.utils.io import Workdir, get_workdir_path
from contextvars import ContextVar


class WorkflowTaskLogger:
    _current_logger = ContextVar("current_logger", default=None)
    _default_instance: Optional["WorkflowTaskLogger"] = None

    def __init__(self, workflow_task=None, log_level: int = logging.INFO, use_stdout: bool = False):
        self.log_level = log_level
        self.use_stdout = use_stdout
        self.file_handler = None
        
        try:
            self.worker = get_worker()
            self.task_key =  self.worker.state.tasks[self.worker.get_current_task()].key
        except ValueError as e:
            if e.args[0] != "No worker found":
                raise e
            self.worker = None
        except Exception as e:
            raise e

        if workflow_task is not None:
            self.task_name = workflow_task.name.lower()
            self.log_file = Workdir.register_file(None, f"{self.task_name}.log")
        else:
            self.task_name = "stdout"
            self.log_file = None
        
        if use_stdout:
            self._setup_stdout_logger()

    def _build_formatter(self) -> logging.Formatter:
        return logging.Formatter(
            "%(name)s: %(asctime)s | %(levelname)s | %(filename)s >>> %(message)s"
        )

    def _format_event_message(self, string: str, log_type: int) -> str:
        record = logging.LogRecord(
            name=self.task_name,
            level=log_type,
            pathname=__file__,
            lineno=0,
            msg=string,
            args=(),
            exc_info=None,
        )
        return self._build_formatter().format(record)

    def _setup_stdout_logger(self):
        self.logger = logging.getLogger(self.task_name)
        self.logger.setLevel(self.log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(self.log_level)
            fmt = self._build_formatter()
            handler.setFormatter(fmt)
            self.logger.addHandler(handler)

    def log(self, string: str, log_type: int = logging.INFO):
        self.logger.log(log_type, string)
        if self.worker is not None:
            self.worker.log_event(
                "runtimes",
                {"task_log": self._format_event_message(string, log_type), "task": self.task_key},
            )

    def get_log_callback(self, log_type=logging.INFO):
        def callback(line):
            self.log(line.rstrip(), log_type)
        return callback

    def __enter__(self):
        self.logger = logging.getLogger(self.task_name)
        self.logger.setLevel(self.log_level)
        fmt = self._build_formatter()
        self.file_handler = logging.FileHandler(self.log_file())
        self.file_handler.setFormatter(fmt)
        self.logger.addHandler(self.file_handler)
        self.start_time = time()

        if self.worker is not None:
            self.log(f"Running task {self.task_name} on dask worker {self.worker.name}")
        else:
            self.log(f"Running task {self.task_name} on local machine")

        self._token = WorkflowTaskLogger._current_logger.set(self)
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time()
        self.log(f"Task {self.task_name} finished after {self.end_time-self.start_time} seconds.")
        if self.file_handler:
            self.logger.removeHandler(self.file_handler)
            self.file_handler.close()
        WorkflowTaskLogger._current_logger.reset(self._token)

    @classmethod
    def _get_default_logger(cls) -> "WorkflowTaskLogger":
        if cls._default_instance is None:
            cls._default_instance = cls(workflow_task=None, use_stdout=True)
        return cls._default_instance

    @classmethod
    def get_current(cls) -> "WorkflowTaskLogger":
        logger = cls._current_logger.get()
        if logger is None:
            return cls._get_default_logger()
        return logger



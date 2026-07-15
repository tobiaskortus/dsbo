import os
import logging
import threading
import traceback
from time import time
from dask.distributed import get_worker
from dsbo.utils.io import DefaultPaths, Workdir
from contextvars import ContextVar


class WorkflowTaskLogger:
    _current_logger = ContextVar("current_logger", default=None)

    def __init__(self, workflow_task=None, log_level: int = logging.INFO, use_stdout: bool = False):
        self.log_level = log_level
        self.file_handler = None

        try:
            self.worker = get_worker()
            self.task_key = self.worker.state.tasks[self.worker.get_current_task()].key
        except ValueError as e:
            if e.args[0] != "No worker found":
                raise
            self.worker = None

        if workflow_task is not None:
            self.task_name = workflow_task.name.lower()
            self.log_file = Workdir.register_file(DefaultPaths.LOGS, f"{self.task_name}.log")
        else:
            self.task_name = "stdout"
            self.log_file = None

        if use_stdout:
            self._setup_stdout_logger()


    def _formatter(self) -> logging.Formatter:
        return logging.Formatter(
            "%(name)s: %(asctime)s | %(levelname)s | %(filename)s >>> %(message)s"
        )

    def _format_event_message(self, string: str, log_type: int) -> str:
        record = logging.LogRecord(
            name=self.task_name, level=log_type, pathname=__file__,
            lineno=0, msg=string, args=(), exc_info=None,
        )
        return self._formatter().format(record)

    def _setup_stdout_logger(self):
        self.logger = logging.getLogger(self.task_name)
        self.logger.setLevel(self.log_level)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(self.log_level)
            handler.setFormatter(self._formatter())
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


    def _pipe_reader(self, read_fd: int, log_type: int):
        with os.fdopen(read_fd, "r", errors="replace") as pipe:
            for line in pipe:
                line = line.rstrip("\r\n")
                if line:
                    self.log(line, log_type)

    def _start_stdio_capture(self):
        self._saved_stdout, self._saved_stderr = os.dup(1), os.dup(2)
        stdout_r, stdout_w = os.pipe()
        stderr_r, stderr_w = os.pipe()
        os.dup2(stdout_w, 1); os.close(stdout_w)
        os.dup2(stderr_w, 2); os.close(stderr_w)
        os.register_at_fork(after_in_child=lambda: (os.dup2(self._saved_stdout, 1), os.dup2(self._saved_stderr, 2)))
        self._stdout_thread = threading.Thread(target=self._pipe_reader, args=(stdout_r, logging.INFO), daemon=True)
        self._stderr_thread = threading.Thread(target=self._pipe_reader, args=(stderr_r, logging.ERROR), daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()

    def _stop_stdio_capture(self):
        os.dup2(self._saved_stdout, 1); os.close(self._saved_stdout)
        os.dup2(self._saved_stderr, 2); os.close(self._saved_stderr)
        self._stdout_thread.join()
        self._stderr_thread.join()


    def __enter__(self):
        self.logger = logging.getLogger(self.task_name)
        self.logger.setLevel(self.log_level)
        self._old_propagate = self.logger.propagate
        self.logger.propagate = False
        self.file_handler = logging.FileHandler(self.log_file())
        self.file_handler.setFormatter(self._formatter())
        self.logger.addHandler(self.file_handler)
        self.start_time = time()

        worker_info = f"dask worker {self.worker.name}" if self.worker is not None else "local machine"
        self.log(f"Running task {self.task_name} on {worker_info}")

        self._token = WorkflowTaskLogger._current_logger.set(self)
        self._start_stdio_capture()
        return self

    def __exit__(self, exc_type, exc_value, tb):
        if exc_type is not None:
            self.log("".join(traceback.format_exception(exc_type, exc_value, tb)), logging.ERROR)

        self._stop_stdio_capture()

        self.end_time = time()
        self.log(f"Task {self.task_name} finished after {self.end_time - self.start_time:.2f}s.")
        if self.file_handler:
            self.logger.removeHandler(self.file_handler)
            self.file_handler.close()
        self.logger.propagate = self._old_propagate
        WorkflowTaskLogger._current_logger.reset(self._token)


    @classmethod
    def get_current(cls) -> "WorkflowTaskLogger":
        logger = cls._current_logger.get()
        if logger is None:
            return cls(workflow_task=None, use_stdout=True)
        return logger



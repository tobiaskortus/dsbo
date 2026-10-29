import os
import shutil
import subprocess
from threading import Thread
from typing import Callable, List, Optional


def available_cores() -> int:
    """Return the number of CPUs available to the current process.

    SLURM exposes the per-task CPU allocation through ``SLURM_CPUS_PER_TASK``;
    otherwise fall back to the process CPU affinity and finally to the total
    number of CPUs on the machine.
    """
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        try:
            return max(1, int(slurm_cpus))
        except ValueError:
            pass

    process_cpu_count = getattr(os, "process_cpu_count", None)
    if process_cpu_count is not None:
        count = process_cpu_count()
        if count:
            return count

    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def which_python():
    for cmd in ("python3", "python"):
        if shutil.which(cmd):
            return cmd
    raise RuntimeError("No python interpreter found.")


def python_script_args(script_path: str, args: List[str]) -> List[str]:
    return [which_python(), script_path] + args


def subprocess_run(cmd: List[str],
                   stdout_callback: Callable[[str], None],
                   stderr_callback: Callable[[str], None]) -> int:
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    def stream(pipe, callback):
        for line in iter(pipe.readline, ''):
            callback(line)
        pipe.close()

    threads = [
        Thread(target=stream, args=(process.stdout, stdout_callback)),
        Thread(target=stream, args=(process.stderr, stderr_callback)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return process.wait()

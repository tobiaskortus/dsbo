import shutil
import subprocess
from threading import Thread
from typing import Callable, List, Optional


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

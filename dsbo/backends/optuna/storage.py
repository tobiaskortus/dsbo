import time
import signal
import subprocess
from redis import ConnectionError, Redis
from typing import Optional


def redis_cmd(port: int, password: Optional[str] = None):
    password_cmd = f" --requirepass {password}" if password != None else ""
    return f"""!/bin/bash
    export LC_ALL=C
    export LANG=C

    apptainer exec docker://redis:latest redis-server --port {port} {password_cmd} &
    """


class RedisApptainerServer:
    def __init__(self, port=6379, password=None, pidfile='redis.pid', timeout=30):
        self.host = "127.0.0.1"
        self.port = port
        self.password = password
        self.pidfile = pidfile
        self.timeout = timeout
        self.proc = None

    def _run_bash_script(self):
        self.proc = subprocess.Popen(['bash', '-c', redis_cmd(self.port, self.password)])

    def _wait_for_redis(self):
        client = Redis(host=self.host, port=self.port, password=self.password)
        start_time = time.time()
        while True:
            try:
                if client.ping():
                    client.close()
                    return
            except ConnectionError:
                pass
            if time.time() - start_time > self.timeout:
                client.close()
                raise TimeoutError(f"Timed out waiting for Redis at {self.host}:{self.port}")
            time.sleep(0.5)

    def _stop_redis(self):
        if self.proc == None:
            return

        if self.proc:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()

    def __enter__(self):
        self._run_bash_script()
        self._wait_for_redis()
        print(f"Redis server is listening now on port {self.port}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Stopping redis server process...")
        self._stop_redis()


def get_journal_redis_backend_url(server: RedisApptainerServer):
    return f'redis://default:{server.password}@{server.host}:{server.port}/db'
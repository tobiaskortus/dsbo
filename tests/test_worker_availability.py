import pytest

from dsbo.worker.dask import DaskWorkerBase, WorkerContext, check_worker_available
from dsbo.worker.spec import LocalWorkerSpec, SLURMWorkerSpec


class UnavailableSpec:
    def is_available(self) -> bool:
        return False


class UnavailableWorker(DaskWorkerBase):
    job_specs = (UnavailableSpec(),)

    def __init__(self):
        pass


class TestWorkerSpecAvailability:
    def test_local_worker_is_available(self):
        assert LocalWorkerSpec.default_local().is_available()

    def test_slurm_availability_follows_required_commands(self, monkeypatch):
        spec = SLURMWorkerSpec(cores=1, processes=1, memory="1GB", walltime="00:10:00")

        monkeypatch.setattr("dsbo.worker.spec.shutil.which", lambda command: f"/usr/bin/{command}")
        assert spec.is_available()

        monkeypatch.setattr("dsbo.worker.spec.shutil.which", lambda command: None)
        assert not spec.is_available()


class TestWorkerContextAvailability:
    def test_unavailable_worker_falls_back_to_local_execution(self):
        worker = UnavailableWorker()
        assert not check_worker_available(worker)

        with pytest.warns(UserWarning, match="not available"):
            with WorkerContext.use_worker(worker):
                assert WorkerContext.get_current_worker() is None

    def test_available_worker_is_used(self):
        worker = object.__new__(DaskWorkerBase)
        assert check_worker_available(worker)

        with WorkerContext.use_worker(worker):
            assert WorkerContext.get_current_worker() is worker

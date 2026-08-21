from __future__ import annotations

import os
import time
import torch
import socket
import logging
from datetime import timedelta
from typing import Any, TYPE_CHECKING

from dsbo.worker.spec import LocalWorkerSpec

if TYPE_CHECKING:
    from dask.distributed import Client
    from dsbo.worker.spec import WorkerSpec


logger = logging.getLogger("dsbo.ddp")

DDP_RENDEZVOUS_TIMEOUT_SECONDS = 60

class DDPDispatchError(RuntimeError):
    """Raised when a DDP dispatch or rendezvous precondition cannot be met."""


class JobQueueHealthError(RuntimeError):
    """Raised when jobqueue workers cannot reach each other for DDP rendezvous."""


def probe_worker(accelerator: str = "auto") -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "hostname": socket.gethostname(),
        "cuda_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
    }


def reserve_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def parse_worker_address(address: str) -> tuple[str, int]:
    """Valid schemes:
    * ``tcp://host:port``
    * ``tcp://host/pid/tid``
    * ``inproc://host/pid/tid``
    """
    base = address.split("://", 1)[-1]
    host = base.split(":")[0].split("/")[0]
    if ":" in base:
        port = base.split(":", 1)[1].split("/")[0]
    else:
        port = base.split("/")[1]
    return host, int(port)


def is_tcp_worker(address: str) -> bool:
    return address.startswith("tcp://")


def probe_peer_reachable(host: str, port: int, timeout: float = 5.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            pass
        return True
    except OSError:
        return False


def choose_backend(accelerator: str, workers_have_cuda: bool) -> str:
    """Choose ``nccl`` (GPU) or ``gloo`` otherwise based on worker-side topology."""
    if accelerator in ("gpu", "cuda") or (accelerator == "auto" and workers_have_cuda):
        return "nccl"
    return "gloo"


def get_worker_info(
    client: Client,
    world_size: int,
    accelerator: str = "auto",
    resource_key: str | None = None,
) -> list[dict[str, Any]]:
    scheduler_info = client.scheduler_info()
    workers = scheduler_info.get("workers", {})
    addresses = sorted(
        address
        for address, metadata in workers.items()
        if resource_key is None
        or metadata.get("resources", {}).get(resource_key, 0) >= 1
    )
    logger.info(
        "ddp setup: %d/%d rank slots available%s (%s)",
        len(addresses), world_size,
        f" for resource {resource_key}" if resource_key else " on cluster",
        ", ".join(addresses),
    )
    if len(addresses) < world_size:
        resource = f" matching resource '{resource_key}'" if resource_key else ""
        raise DDPDispatchError(
            f"DDP dispatch requested {world_size} rank slots but only "
            f"{len(addresses)} dask workers{resource} are available. Scale the cluster "
            f"to at least {world_size} workers so each DDP rank "
            "gets a distinct worker slot."
        )

    addresses = addresses[:world_size]

    t0 = time.monotonic()
    logger.info("ddp setup: probing %d worker(s) for host/GPU topology...", len(addresses))
    probes = client.run(probe_worker, workers=addresses, wait=True)
    logger.info("ddp setup: worker probe done (%.2fs)", time.monotonic() - t0)

    master_address = addresses[0]
    master_host, master_worker_port = parse_worker_address(master_address)
    logger.info("ddp setup: master worker = %s (host %s)", master_address, master_host)

    t0 = time.monotonic()
    master_port = client.run(reserve_port, workers=[master_address], wait=True)[master_address]
    logger.info("ddp setup: reserved rendezvous port %s:%d (%.2fs)", master_host, master_port, time.monotonic() - t0)

    backend = choose_backend(accelerator, any(p.get("cuda_device_count", 0) > 0 for p in probes.values()))
    logger.info("ddp setup: backend=%s accelerator=%s", backend, accelerator)

    peer_address = addresses[1] if len(addresses) > 1 else master_address

    if (
        peer_address != master_address
        and is_tcp_worker(master_address)
        and is_tcp_worker(peer_address)
    ):
        t0 = time.monotonic()
        logger.info("ddp setup: checking peer %s can reach master %s...", peer_address, master_address)
        reachable = client.run(
            probe_peer_reachable,
            master_host,
            master_worker_port,
            workers=[peer_address],
            wait=True,
        ).get(peer_address)
        logger.info("ddp setup: peer reachability check done (%.2fs, reachable=%s)", time.monotonic() - t0, reachable)
        if not reachable:
            raise JobQueueHealthError(
                f"DDP peer worker {peer_address} cannot reach the rendezvous "
                f"host {master_host}:{master_worker_port}. Jobqueue nodes must "
                "reach the master node over the network; check proxy/NAT/container "
                "isolation or the cluster interface configuration."
            )
    elif not is_tcp_worker(master_address):
        logger.debug(
            "skipping peer reachability probe for non-TCP transport %s",
            master_address,
        )

    local_counter: dict[str, int] = {}
    infos: list[dict[str, Any]] = []
    for global_rank, address in enumerate(addresses):
        probe = probes[address]
        host = probe.get("hostname", master_host)
        local_rank = local_counter.get(host, 0)
        local_counter[host] = local_rank + 1
        infos.append(
            {
                "rank": global_rank,
                "local_rank": local_rank,
                "world_size": world_size,
                "master_addr": master_host,
                "master_port": master_port,
                "worker": address,
                "backend": backend,
                "cuda_device_count": probe.get("cuda_device_count", 0),
            }
        )
    return infos


def assert_ddp_supported(spec: WorkerSpec, world_size: int | None = None) -> None:
    if world_size is not None and world_size < 1:
        raise ValueError(f"world_size must be >= 1, got {world_size}.")

    if isinstance(spec, LocalWorkerSpec):
        raise DDPDispatchError(
            "DDP training requires a non-local jobqueue worker spec such as "
            "SLURMWorkerSpec; LocalWorkerSpec DDP is not supported."
        )


def configure_rank_env(info: dict[str, Any], n_gpus_per_rank: int = 1) -> str:
    os.environ["MASTER_ADDR"] = info["master_addr"]
    os.environ["MASTER_PORT"] = str(info["master_port"])
    os.environ["RANK"] = str(info["rank"])
    os.environ["LOCAL_RANK"] = str(info["local_rank"])
    os.environ["WORLD_SIZE"] = str(info["world_size"])

    backend = info["backend"]
    logger.info(
        "rank %s: rank env set (MASTER=%s:%s, RANK=%s, LOCAL_RANK=%s, WORLD_SIZE=%s, backend=%s)",
        info["rank"], info["master_addr"], info["master_port"],
        info["rank"], info["local_rank"], info["world_size"], backend,
    )
    if backend == "nccl":
        device_count = info.get("cuda_device_count", 0)
        if device_count > 0:
            gpus = range(min(n_gpus_per_rank, device_count))
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    return backend


def init_rendezvous(info: dict[str, Any], timeout_s: int = DDP_RENDEZVOUS_TIMEOUT_SECONDS) -> None:
    import torch.distributed as dist

    backend = info["backend"]
    start = time.monotonic()
    try:
        logger.info(
            "rank %s: initializing process group (backend=%s, master=%s:%s)",
            info["rank"], backend, info["master_addr"], info["master_port"],
        )
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(seconds=timeout_s),
        )
    except (TimeoutError, OSError, dist.DistNetworkError, dist.DistStoreError) as error:
        if dist.is_initialized():
            dist.destroy_process_group()
        raise JobQueueHealthError(
            f"DDP rank {info['rank']}: process group rendezvous failed on "
            f"{info['master_addr']}:{info['master_port']} after {timeout_s}s. "
            "Ensure the jobqueue allocation starts all ranks together and that nodes "
            "can reach the master node (check firewall / container-network config)."
        ) from error

    logger.info(
        "rank %s: process group initialized (world_size=%s) rendezvous took %.2fs",
        info["rank"], info["world_size"], time.monotonic() - start,
    )

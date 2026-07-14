import logging
import os
import shutil
import subprocess
import warnings
from pathlib import Path
from enum import Enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Type, Union, TYPE_CHECKING

from dsbo.config import config
from dsbo.logger import WorkflowTaskLogger
from dsbo.utils.process import subprocess_run

if TYPE_CHECKING:
    from dsbo.workflow import Workflow


@dataclass(frozen=True)
class ImageRef:
    """A reference to a container image in a registry (e.g. Docker Hub).
    Can be pulled directly by Docker, or converted to a .sif file by Apptainer/Singularity.
    """
    uri: str

    def __hash__(self) -> int:
        return hash(self.uri)

    def _normalize_uri(self, uri: str) -> str:
        for prefix in ("https://", "http://", "docker://"):
            if uri.startswith(prefix):
                uri = uri[len(prefix):]
                break
        result = []
        for c in uri:
            if c.isalnum() or c in "._-":
                result.append(c)
            else:
                result.append("_")
        return "".join(result)

    def get_image_file_path(self, container_path: Optional[str] = None) -> Path:
        if container_path is None:
            container_path = config.get("container.path", os.getcwd())
        normalized = self._normalize_uri(self.uri)
        return Path(container_path, f"{normalized}.sif")


@dataclass(frozen=True)
class ImageFile:
    """A local .sif container image file, used with Apptainer/Singularity."""
    path: str | Path

    def __post_init__(self) -> None:
        if isinstance(self.path, str):
            object.__setattr__(self, "path", Path(self.path))

    def __hash__(self) -> int:
        return hash(self.path.resolve())


@dataclass(frozen=True)
class DefinitionFile:
    """A container definition file (Dockerfile or Singularity .def) 
    that builds the container image on first use.
    """
    path: str | Path
    container_name: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.path, str):
            object.__setattr__(self, "path", Path(self.path))

    def __hash__(self) -> int:
        return hash((self.path.resolve(), self.container_name))

    def _def_type(self) -> str:
        p = str(self.path).lower()
        if "dockerfile" in p:
            return "docker"
        if p.endswith(".def"):
            return "singularity"
        return "docker"


def discover_container_environments() -> Dict[str, bool]:
    return {
        "docker": shutil.which("docker") is not None,
        "apptainer": shutil.which("apptainer") is not None,
        "singularity": shutil.which("singularity") is not None,
    }


class ImageRequirement(Enum):
    SINGULARITY_ONLY = "singularity_only"
    DOCKER_ONLY = "docker_only"
    EITHER = "either"


class ContainerType(Enum):
    SINGULARITY = "singularity"
    APPTAINER = "apptainer"
    DOCKER = "docker"


def image_type_required(image_def: ImageRef | ImageFile | DefinitionFile) -> ImageRequirement:
    if isinstance(image_def, ImageFile):
        return ImageRequirement.SINGULARITY_ONLY
    if isinstance(image_def, ImageRef):
        return ImageRequirement.EITHER
    if isinstance(image_def, DefinitionFile):
        dtype = image_def._def_type()
        if dtype == "singularity":
            return ImageRequirement.SINGULARITY_ONLY
        return ImageRequirement.DOCKER_ONLY
    raise Exception("Invalid image definition provided.")


def select_compatible_environment(required: ImageRequirement, available: Dict[str, bool]) -> Optional[ContainerType]:
    for ct in (ContainerType.APPTAINER, ContainerType.SINGULARITY, ContainerType.DOCKER):
        if not available.get(ct.value):
            continue
        if required == ImageRequirement.SINGULARITY_ONLY and ct == ContainerType.DOCKER:
            continue
        if required == ImageRequirement.DOCKER_ONLY and ct != ContainerType.DOCKER:
            continue
        return ct
    raise Exception(f"The provided image definition reuires {required}")


def check_mismatch(image_def: Union[ImageRef, ImageFile, DefinitionFile], selected_env: ContainerType) -> None:
    if isinstance(image_def, ImageFile) and selected_env == ContainerType.DOCKER:
        raise RuntimeError(
            "ImageFile (.sif) used with Docker environment — "
            ".sif files are not runnable by Docker. "
            "Use Apptainer/Singularity instead."
        )
    if isinstance(image_def, DefinitionFile):
        dtype = image_def._def_type()
        if dtype == "docker" and selected_env != ContainerType.DOCKER:
            raise RuntimeError(
                f"ImageDefinitionFile (Dockerfile) used with {selected_env} — "
                "Dockerfiles cannot be built by Apptainer/Singularity."
            )
        if dtype == "singularity" and selected_env == ContainerType.DOCKER:
            raise RuntimeError(
                "ImageDefinitionFile (.def) used with Docker — "
                "Singularity definition files cannot be built by Docker."
            )



class ImageProvider(ABC):
    @abstractmethod
    def exists(self, image: str) -> bool:
        pass

    @abstractmethod
    def pull(self, image: str) -> bool:
        pass

    @abstractmethod
    def build(self, def_file: DefinitionFile, output_dir: str = "") -> str:
        pass

    @abstractmethod
    def resolve_path(self, image: str) -> str:
        pass

    @abstractmethod
    def prepare_image(self, image_def: ImageRef | ImageFile | DefinitionFile,
                      output_dir: str = "") -> str:
        """Ensure the image is available locally, returning the resolved image reference."""
        pass


def _normalize_path_tag(path_str: str) -> str:
    stem = Path(path_str).stem
    result = []
    for c in stem.lower():
        if c.isalnum() or c in "._-":
            result.append(c)
        else:
            result.append("_")
    return "".join(result)


class DockerImageProvider(ImageProvider):
    def __init__(self) -> None:
        self._executable: Optional[str] = None

    def _get_executable(self) -> str:
        if self._executable is None:
            for cmd in ("docker",):
                if shutil.which(cmd):
                    self._executable = cmd
                    return self._executable
            raise RuntimeError("Docker executable not found on PATH.")
        return self._executable

    def exists(self, image: str) -> bool:
        cmd = [self._get_executable(), "images", "--format", "{{.Repository}}:{{.Tag}}", image]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return image in result.stdout

    def pull(self, image: str) -> bool:
        cmd = [self._get_executable(), "pull", image]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0

    def resolve_path(self, image: str) -> str:
        return image

    def build(self, def_file: DefinitionFile, output_dir: str = "") -> str:
        def_path = Path(str(def_file.path))
        tag = def_file.container_name or f"dsbo_{_normalize_path_tag(str(def_path))}"
        if self.exists(tag):
            return tag
        context_dir = str(def_path.parent)
        if not shutil.which("docker"):
            raise RuntimeError("Docker executable not found for building Dockerfile.")
        cmd = ["docker", "build", "-t", tag, context_dir]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to build Dockerfile {def_path} (exit code {result.returncode})\n"
                f"{result.stderr.strip()}"
            )
        return tag

    def prepare_image(self, image_def: ImageRef | ImageFile | DefinitionFile,
                      output_dir: str = "") -> str:
        if isinstance(image_def, ImageRef):
            return image_def.uri
        if isinstance(image_def, ImageFile):
            raise RuntimeError(
                "ImageFile (.sif) used with Docker environment — "
                ".sif files are not runnable by Docker. "
                "Use Apptainer/Singularity instead."
            )
        if isinstance(image_def, DefinitionFile):
            return self.build(image_def, output_dir)
        raise Exception("Invalid image definition provided.")


class SingularityImageProvider(ImageProvider):
    def __init__(self, image_dir: Optional[str] = None) -> None:
        self._image_dir = Path(image_dir) if image_dir else None
        self._executable: Optional[str] = None

    def _get_executable(self) -> str:
        if self._executable is None:
            for cmd in ("singularity", "apptainer"):
                if shutil.which(cmd):
                    self._executable = cmd
                    return self._executable
            raise RuntimeError("No Singularity/Apptainer executable found on PATH.")
        return self._executable

    def _image_path(self, image: str) -> Path:
        if self._image_dir:
            return self._image_dir / image
        return Path(image)

    def exists(self, image: str) -> bool:
        return self._image_path(image).exists()

    def pull(self, image: str) -> bool:
        path = self._image_path(image)
        path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [self._get_executable(), "pull", str(path), image]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return result.returncode == 0

    def resolve_path(self, image: str) -> str:
        return str(self._image_path(image))

    def build(self, def_file: DefinitionFile, output_dir: str = "") -> str:
        def_path = Path(str(def_file.path))
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        sif_name = def_file.container_name or f"{_normalize_path_tag(str(def_path))}.sif"
        output_sif = output_dir_path / (sif_name if sif_name.endswith('.sif') else f"{sif_name}.sif")

        if output_sif.exists():
            return str(output_sif)

        executable = None
        for cmd in ("apptainer", "singularity"):
            if shutil.which(cmd):
                executable = cmd
                break
        if executable is None:
            raise RuntimeError("No Apptainer/Singularity executable found for building .def file.")

        cmd = [executable, "build", str(output_sif), str(def_path)]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(def_path.parent))
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to build .def file {def_path} to {output_sif} "
                f"(exit code {result.returncode})\n"
                f"{result.stderr.strip()}"
            )
        return str(output_sif)

    def prepare_image(self, image_def: ImageRef | ImageFile | DefinitionFile,
                      output_dir: str = "") -> str:
        if isinstance(image_def, ImageRef):
            return self._pull_ref_to_sif(image_def, output_dir)
        if isinstance(image_def, ImageFile):
            return self.resolve_path(str(image_def.path))
        if isinstance(image_def, DefinitionFile):
            return self.build(image_def, output_dir)
        raise Exception("Invalid image definition provided.")

    def _ensure_docker_scheme(self, uri: str) -> str:
        for prefix in ("docker://", "https://", "http://"):
            if uri.startswith(prefix):
                return uri
        return f"docker://{uri}"

    def _pull_ref_to_sif(self, image_ref: ImageRef, output_dir: str) -> str:
        sif_path = image_ref.get_image_file_path(output_dir)
        sif_path.parent.mkdir(parents=True, exist_ok=True)

        if sif_path.exists():
            return str(sif_path)

        pull_uri = self._ensure_docker_scheme(image_ref.uri)
        cmd = [self._get_executable(), "pull", str(sif_path), pull_uri]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to pull {image_ref.uri} to {sif_path} "
                f"(exit code {result.returncode})\n"
                f"{result.stderr.strip()}"
            )
        return str(sif_path)


@dataclass(frozen=True)
class ContainerConfig:
    cuda_enabled: bool = False
    bind_mounts: tuple[tuple[str, str], ...] = ()
    extra_flags: tuple[str, ...] = ()

    def enable_cuda(self) -> "ContainerConfig":
        if self.cuda_enabled:
            return self
        return type(self)(cuda_enabled=True,
                          bind_mounts=self.bind_mounts,
                          extra_flags=self.extra_flags)

    def with_extra_flags(self, *flags: str) -> "ContainerConfig":
        return type(self)(cuda_enabled=self.cuda_enabled,
                          bind_mounts=self.bind_mounts,
                          extra_flags=self.extra_flags + flags)

    def with_bind_mount(self, *mounts: tuple[str, str]) -> "ContainerConfig":
        return type(self)(cuda_enabled=self.cuda_enabled,
                          bind_mounts=self.bind_mounts + mounts,
                          extra_flags=self.extra_flags)


class ContainerRuntime(ABC):
    def __init__(self, image: str, image_provider: ImageProvider,
                 config: ContainerConfig) -> None:
        self._image = image
        self._image_provider = image_provider
        self._config = config

    @abstractmethod
    def run(self, args: List[str]) -> int:
        pass

    @abstractmethod
    def execute(self, args: List[str]) -> int:
        pass

    @abstractmethod
    def build_exec_command(self, image: str, args: List[str]) -> List[str]:
        pass

    @abstractmethod
    def build_run_command(self, image: str, args: List[str]) -> List[str]:
        pass

    def _cuda_flags(self) -> List[str]:
        if self._config.cuda_enabled:
            return self._backend_cuda_flags()
        return []

    @abstractmethod
    def _backend_cuda_flags(self) -> List[str]:
        pass

    def _bind_mount_flags(self) -> List[str]:
        flags = []
        for src, dst in self._config.bind_mounts:
            flags.extend(self._backend_bind_mount_flag(src, dst))
        return flags

    @abstractmethod
    def _backend_bind_mount_flag(self, src: str, dst: str) -> List[str]:
        pass

    def _resolve_image(self, image: str) -> str:
        return self._image_provider.resolve_path(image)

    def _get_executable(self) -> str:
        return self._image_provider._get_executable()  # type: ignore[attr-defined]

    def _build_command_prefix(self, command: str, image: str) -> List[str]:
        executable = self._get_executable()
        return [executable, command, *self._cuda_flags(),
                *self._bind_mount_flags(),
                *list(self._config.extra_flags), self._resolve_image(image)]


class DockerContainerRuntime(ContainerRuntime):
    def __init__(self, image: str, image_provider: ImageProvider,
                 config: ContainerConfig) -> None:
        super().__init__(image, image_provider, config)

    def _backend_cuda_flags(self) -> List[str]:
        return ["--gpus all"]

    def _backend_bind_mount_flag(self, src: str, dst: str) -> List[str]:
        return ["-v", f"{src}:{dst}"]

    def build_run_command(self, image: str, args: List[str]) -> List[str]:
        prefix = self._build_command_prefix("run", image)
        return prefix[:-1] + ["--rm", prefix[-1]] + args

    def build_exec_command(self, image: str, args: List[str]) -> List[str]:
        # Docker exec requires a running container, so use `docker run --rm`
        # instead to start a fresh container with the command as arguments
        prefix = self._build_command_prefix("run", image)
        return prefix[:-1] + ["--rm", prefix[-1]] + args

    def run(self, args: List[str]) -> int:
        cmd = self.build_run_command(self._image, args)
        result = subprocess.run(cmd)
        return result.returncode

    def execute(self, args: List[str]) -> int:
        cmd = self.build_exec_command(self._image, args)
        result = subprocess.run(cmd)
        return result.returncode


class SingularityContainerRuntime(ContainerRuntime):
    def __init__(self, image: str, image_provider: ImageProvider,
                 config: ContainerConfig) -> None:
        super().__init__(image, image_provider, config)

    def _backend_cuda_flags(self) -> List[str]:
        return ["--nv"]

    def _backend_bind_mount_flag(self, src: str, dst: str) -> List[str]:
        return ["--bind", f"{src}:{dst}"]

    def build_run_command(self, image: str, args: List[str]) -> List[str]:
        return self._build_command_prefix("run", image) + args

    def build_exec_command(self, image: str, args: List[str]) -> List[str]:
        return self._build_command_prefix("exec", image) + args

    def run(self, args: List[str]) -> int:
        cmd = self.build_run_command(self._image, args)
        result = subprocess.run(cmd)
        return result.returncode

    def execute(self, args: List[str]) -> int:
        cmd = self.build_exec_command(self._image, args)
        result = subprocess.run(cmd)
        return result.returncode


_container_runtime_registry: Dict[ContainerType, Type[ContainerRuntime]] = {
    ContainerType.SINGULARITY: SingularityContainerRuntime,
    ContainerType.APPTAINER: SingularityContainerRuntime,
    ContainerType.DOCKER: DockerContainerRuntime,
}


def make_provider(container_type: ContainerType,
                   image_dir: Optional[str] = None) -> ImageProvider:
    if container_type == ContainerType.DOCKER:
        return DockerImageProvider()
    return SingularityImageProvider(image_dir)


def get_runtime_for_type(container_type: ContainerType, image: str,
                          image_provider: ImageProvider,
                          config: ContainerConfig) -> ContainerRuntime:
    cls = _container_runtime_registry.get(container_type)
    if cls is None:
        raise ValueError(f"Unknown container type: {container_type}")
    return cls(image, image_provider, config)




def warm_container_dependencies(workflow: "Workflow", verbose: bool = True) -> dict[ImageRef | ImageFile | DefinitionFile, str]:
    from tqdm import tqdm

    result = dict.fromkeys(
        n._image_def for n in workflow.task_graph.nodes
        if isinstance(n, ContainerWrapperBase)
    )

    for d in tqdm(result, desc="Warming containers", unit="container", disable=not verbose):
        env = select_compatible_environment(image_type_required(d), discover_container_environments())
        check_mismatch(d, env)
        img_dir = config.get("container.path", os.getcwd()) if env != ContainerType.DOCKER else None
        result[d] = make_provider(env, img_dir).prepare_image(d, img_dir or "")

    return result


class ContainerWrapperBase:
    def __init__(self, container_path: ImageRef | ImageFile | DefinitionFile,
                 container_extra_flags: Optional[str] = None,
                 config: Optional[ContainerConfig] = None,
                 cuda_enabled: bool = False,
                 verbose: bool = True,
                 **kwargs) -> None:

        container_type = kwargs.pop("container_type", None)
        if container_type is not None:
            warnings.warn(
                "container_type is deprecated; backend is now auto-detected from image path.",
                DeprecationWarning, stacklevel=3,
            )

        super().__init__(**kwargs)
        self._verbose = verbose
        self._runtime: Optional[ContainerRuntime] = None
        self._image_def = container_path
        self._container_path: str = ""
        self._detected_type = container_type if container_type is not None else ContainerType.DOCKER

        if config is None:
            extra = container_extra_flags.split() if container_extra_flags else ()
            config = ContainerConfig(cuda_enabled=cuda_enabled,
                                     extra_flags=tuple(extra))
        self._config = config

    def _make_provider(self, container_type: ContainerType,
                   image_dir: Optional[str] = None) -> ImageProvider:
        return make_provider(container_type, image_dir)

    def _get_runtime_for_type(self,
                              container_type: ContainerType, image: str,
                              image_provider: ImageProvider,
                              config: ContainerConfig) -> ContainerRuntime:
        return get_runtime_for_type(container_type, image, image_provider, config)

    def _ensure_runtime(self) -> ContainerRuntime:
        if self._runtime is not None:
            return self._runtime

        available = discover_container_environments()
        required = image_type_required(self._image_def)
        selected_env = select_compatible_environment(required, available)
        check_mismatch(self._image_def, selected_env)

        self._detected_type = selected_env

        image_dir = config.get("container.path", os.getcwd()) if selected_env != ContainerType.DOCKER else None
        provider = self._make_provider(selected_env, image_dir)

        image_ref = provider.prepare_image(self._image_def, image_dir or "")

        self._runtime = self._get_runtime_for_type(selected_env, image_ref, provider, self._config)
        self._container_path = image_ref
        return self._runtime

    def __construct_cmd(self, args: List[str], is_exec: bool = False) -> List[str]:
        runtime = self._ensure_runtime()
        if is_exec:
            return runtime.build_exec_command(self._container_path, args)
        return runtime.build_run_command(self._container_path, args)

    def _run_with_logging(self, cmd: List[str]) -> None:
        logger = WorkflowTaskLogger.get_current()

        if logger is not None:
            logger.log(f"Running container: {self._container_path} with args: {' '.join(cmd)}")
            stdout = logger.get_log_callback(logging.INFO)
            stderr = logger.get_log_callback(logging.ERROR)
        else:
            stdout = lambda x: print(x)
            stderr = lambda x: print(x)

        process_returncode = subprocess_run(cmd, stdout, stderr)
        if process_returncode != 0 and logger is not None:
            logger.log(f"Container process failed with return code {process_returncode}",
                       log_type=logging.ERROR)

    def run_container(self, args: List[str]) -> None:
        self._run_with_logging(self.__construct_cmd(args, is_exec=False))

    def exec_container(self, args: List[str]) -> None:
        self._run_with_logging(self.__construct_cmd(args, is_exec=True))

    def bind_mount(self, *mounts: tuple[str, str]) -> None:
        self._config = self._config.with_bind_mount(*mounts)

    @property
    def container_path(self) -> str:
        return self._container_path

    @property
    def container_extra_flags(self) -> Optional[str]:
        if self._config.extra_flags:
            return " ".join(self._config.extra_flags)
        return None

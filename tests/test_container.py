import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from dsbo.workflow import Workflow
from dsbo.modules import Module
from dsbo.modules.dsl import DependencyDict
from dsbo.worker.spec import LocalWorkerSpec, SLURMWorkerSpec

from dsbo.container import (
    ContainerConfig,
    ContainerType,
    ContainerWrapperBase,
    DockerContainerRuntime,
    DockerImageProvider,
    DefinitionFile,
    ImageFile,
    ImageProvider,
    ImageRef,
    ImageRequirement,
    SingularityContainerRuntime,
    SingularityImageProvider,
    check_mismatch,

    image_type_required,
    make_provider as make_provider,
    _normalize_path_tag,
    get_runtime_for_type,
    select_compatible_environment,
    discover_container_environments,
    warm_container_dependencies,

    subprocess_run,
)



class TestContainerConfig:
    def test_default_config(self):
        config = ContainerConfig()
        assert config.cuda_enabled is False
        assert config.bind_mounts == ()
        assert config.extra_flags == ()

    def test_custom_config(self):
        config = ContainerConfig(
            cuda_enabled=True,
            bind_mounts=(("/host", "/container"),),
            extra_flags=("--rm", "-v /foo:/bar"),
        )
        assert config.cuda_enabled is True
        assert config.bind_mounts == (("/host", "/container"),)
        assert config.extra_flags == ("--rm", "-v /foo:/bar")

    def test_frozen(self):
        config = ContainerConfig(cuda_enabled=False)
        with pytest.raises(AttributeError):
            config.cuda_enabled = True
        with pytest.raises(AttributeError):
            config.bind_mounts = (("/src", "/dst"),)
        with pytest.raises(AttributeError):
            config.extra_flags = ("--rm",)

    def test_enable_cuda_returns_new_instance(self):
        config = ContainerConfig()
        new_config = config.enable_cuda()
        assert new_config is not config
        assert new_config.cuda_enabled is True
        assert config.cuda_enabled is False

    def test_enable_cuda_idempotent(self):
        config = ContainerConfig(cuda_enabled=True)
        same = config.enable_cuda()
        assert same is config

    def test_enable_cuda_preserves_bind_mounts_and_extra_flags(self):
        config = ContainerConfig(
            bind_mounts=(("/src", "/dst"),),
            extra_flags=("--rm",),
        )
        new_config = config.enable_cuda()
        assert new_config.cuda_enabled is True
        assert new_config.bind_mounts == (("/src", "/dst"),)
        assert new_config.extra_flags == ("--rm",)

    def test_with_extra_flags_returns_new_instance(self):
        config = ContainerConfig()
        new_config = config.with_extra_flags("--gpus all", "--rm")
        assert new_config is not config
        assert new_config.extra_flags == ("--gpus all", "--rm")
        assert config.extra_flags == ()

    def test_with_extra_flags_appends(self):
        config = ContainerConfig(extra_flags=("--rm",))
        new_config = config.with_extra_flags("-v /foo:/bar")
        assert new_config.extra_flags == ("--rm", "-v /foo:/bar")

    def test_with_extra_flags_preserves_cuda(self):
        config = ContainerConfig(cuda_enabled=True)
        new_config = config.with_extra_flags("-v /foo:/bar")
        assert new_config.cuda_enabled is True
        assert new_config.extra_flags == ("-v /foo:/bar",)

    def test_with_extra_flags_preserves_bind_mounts(self):
        config = ContainerConfig(bind_mounts=(("/host", "/container"),))
        new_config = config.with_extra_flags("--rm")
        assert new_config.bind_mounts == (("/host", "/container"),)
        assert new_config.extra_flags == ("--rm",)

    def test_with_bind_mount_returns_new_instance(self):
        config = ContainerConfig()
        new_config = config.with_bind_mount(("/src", "/dst"))
        assert new_config is not config
        assert new_config.bind_mounts == (("/src", "/dst"),)
        assert config.bind_mounts == ()

    def test_with_bind_mount_appends(self):
        config = ContainerConfig(bind_mounts=(("/a", "/b"),))
        new_config = config.with_bind_mount(("/c", "/d"))
        assert new_config.bind_mounts == (("/a", "/b"), ("/c", "/d"))

    def test_with_bind_mount_preserves_cuda(self):
        config = ContainerConfig(cuda_enabled=True)
        new_config = config.with_bind_mount(("/src", "/dst"))
        assert new_config.cuda_enabled is True
        assert new_config.bind_mounts == (("/src", "/dst"),)

    def test_with_bind_mount_preserves_extra_flags(self):
        config = ContainerConfig(extra_flags=("--rm",))
        new_config = config.with_bind_mount(("/src", "/dst"))
        assert new_config.extra_flags == ("--rm",)
        assert new_config.bind_mounts == (("/src", "/dst"),)

    def test_chaining(self):
        config = ContainerConfig()
        result = config.enable_cuda().with_extra_flags("--gpus all").with_bind_mount(("/src", "/dst"))
        assert result.cuda_enabled is True
        assert result.extra_flags == ("--gpus all",)
        assert result.bind_mounts == (("/src", "/dst"),)
        # original unchanged
        assert config.cuda_enabled is False
        assert config.extra_flags == ()
        assert config.bind_mounts == ()



class TestDockerImageProvider:
    @patch("shutil.which", return_value="/usr/bin/docker")
    def test_find_executable(self, mock_which):
        provider = DockerImageProvider()
        assert provider._executable is None 
        assert provider._get_executable() == "docker"
        assert provider._executable == "docker"

    @patch("shutil.which", return_value=None)
    def test_find_executable_missing(self, mock_which):
        provider = DockerImageProvider()
        with pytest.raises(RuntimeError, match="Docker executable not found"):
            provider._get_executable()

    @patch("shutil.which", return_value="/usr/bin/docker")
    @patch("subprocess.run")
    def test_exists_found(self, mock_run, mock_which):
        mock_run.return_value = MagicMock(stdout="ubuntu:22.04\n")
        provider = DockerImageProvider()
        assert provider.exists("ubuntu:22.04") is True
        mock_run.assert_called_with(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}", "ubuntu:22.04"],
            capture_output=True, text=True,
        )

    @patch("shutil.which", return_value="/usr/bin/docker")
    @patch("subprocess.run")
    def test_exists_not_found(self, mock_run, mock_which):
        mock_run.return_value = MagicMock(stdout="")
        provider = DockerImageProvider()
        assert provider.exists("nonexistent:tag") is False

    @patch("shutil.which", return_value="/usr/bin/docker")
    @patch("subprocess.run")
    def test_pull_success(self, mock_run, mock_which):
        mock_run.return_value = MagicMock(returncode=0)
        provider = DockerImageProvider()
        assert provider.pull("ubuntu:22.04") is True
        mock_run.assert_called_with(["docker", "pull", "ubuntu:22.04"], capture_output=True, text=True)

    @patch("shutil.which", return_value="/usr/bin/docker")
    @patch("subprocess.run")
    def test_pull_failure(self, mock_run, mock_which):
        mock_run.return_value = MagicMock(returncode=1)
        provider = DockerImageProvider()
        assert provider.pull("ubuntu:22.04") is False

    @patch("shutil.which", return_value="/usr/bin/docker")
    def test_resolve_path(self, mock_which):
        provider = DockerImageProvider()
        assert provider.resolve_path("ubuntu:22.04") == "ubuntu:22.04"

    @patch("shutil.which", return_value="/usr/bin/docker")
    @patch("subprocess.run")
    def test_build_default_tag(self, mock_run, mock_which):
        mock_run.return_value.returncode = 0
        provider = DockerImageProvider()
        def_file = DefinitionFile(path="/path/to/Dockerfile")
        result = provider.build(def_file, "")
        assert result == "dsbo_dockerfile"
        mock_run.assert_called_with(
            ["docker", "build", "-t", "dsbo_dockerfile", "/path/to"],
            capture_output=True, text=True,
        )

    @patch("shutil.which", return_value="/usr/bin/docker")
    @patch("subprocess.run")
    def test_build_with_container_name(self, mock_run, mock_which):
        mock_run.return_value.returncode = 0
        provider = DockerImageProvider()
        def_file = DefinitionFile(path="/path/to/Dockerfile", container_name="my-image")
        result = provider.build(def_file, "")
        assert result == "my-image"
        mock_run.assert_called_with(
            ["docker", "build", "-t", "my-image", "/path/to"],
            capture_output=True, text=True,
        )

    @patch("shutil.which", return_value="/usr/bin/docker")
    @patch("subprocess.run")
    def test_build_failure(self, mock_run, mock_which):
        mock_run.return_value.returncode = 1
        provider = DockerImageProvider()
        def_file = DefinitionFile(path="/path/to/Dockerfile")
        with pytest.raises(RuntimeError, match="Failed to build Dockerfile"):
            provider.build(def_file, "")

    @patch("shutil.which", return_value=None)
    def test_build_no_executable(self, mock_which):
        provider = DockerImageProvider()
        def_file = DefinitionFile(path="/path/to/Dockerfile")
        with pytest.raises(RuntimeError, match="Docker executable not found"):
            provider.build(def_file, "")



class TestSingularityImageProvider:
    @patch("shutil.which", return_value="/usr/bin/singularity")
    def test_find_executable(self, mock_which):
        provider = SingularityImageProvider(image_dir="/path/to/images")
        assert provider._executable is None 
        assert provider._get_executable() == "singularity"
        assert provider._executable == "singularity"

    @patch("shutil.which", return_value=None)
    def test_find_executable_missing(self, mock_which):
        provider = SingularityImageProvider(image_dir="/path/to/images")
        with pytest.raises(RuntimeError, match="No Singularity/Apptainer executable found"):
            provider._get_executable()

    @patch("shutil.which", return_value="/usr/bin/singularity")
    def test_image_path_with_dir(self, mock_which):
        provider = SingularityImageProvider(image_dir="/path/to/images")
        assert str(provider._image_path("myimage.sif")) == "/path/to/images/myimage.sif"

    @patch("shutil.which", return_value="/usr/bin/singularity")
    def test_image_path_without_dir(self, mock_which):
        provider = SingularityImageProvider()
        assert str(provider._image_path("./myimage.sif")) == "myimage.sif"

    @patch("shutil.which", return_value="/usr/bin/singularity")
    @patch("pathlib.Path.exists", return_value=True)
    def test_exists_true(self, mock_exists, mock_which):
        provider = SingularityImageProvider(image_dir="/path/to/images")
        assert provider.exists("myimage.sif") is True

    @patch("shutil.which", return_value="/usr/bin/singularity")
    @patch("pathlib.Path.exists", return_value=False)
    def test_exists_false(self, mock_exists, mock_which):
        provider = SingularityImageProvider(image_dir="/path/to/images")
        assert provider.exists("myimage.sif") is False

    @patch("shutil.which", return_value="/usr/bin/singularity")
    @patch("subprocess.run")
    @patch("pathlib.Path.mkdir")
    def test_pull_success(self, mock_mkdir, mock_run, mock_which):
        mock_run.return_value = MagicMock(returncode=0)
        provider = SingularityImageProvider(image_dir="/path/to/images")
        assert provider.pull("myimage.sif") is True
        mock_run.assert_called_with(
            ["singularity", "pull", "/path/to/images/myimage.sif", "myimage.sif"],
            capture_output=True, text=True,
        )

    @patch("shutil.which", return_value="/usr/bin/singularity")
    @patch("subprocess.run")
    @patch("pathlib.Path.mkdir")
    def test_pull_failure(self, mock_mkdir, mock_run, mock_which):
        mock_run.return_value = MagicMock(returncode=1)
        provider = SingularityImageProvider(image_dir="/path/to/images")
        assert provider.pull("myimage.sif") is False

    @patch("shutil.which", return_value="/usr/bin/singularity")
    def test_resolve_path_with_dir(self, mock_which):
        provider = SingularityImageProvider(image_dir="/path/to/images")
        assert provider.resolve_path("myimage.sif") == "/path/to/images/myimage.sif"

    @patch("shutil.which", return_value="/usr/bin/singularity")
    def test_resolve_path_without_dir(self, mock_which):
        provider = SingularityImageProvider()
        assert provider.resolve_path("./myimage.sif") == "myimage.sif"

    @patch("shutil.which", side_effect=lambda x: "/usr/bin/" + x if x == "apptainer" else None)
    @patch("subprocess.run")
    @patch("pathlib.Path.exists", return_value=False)
    @patch("pathlib.Path.mkdir")
    def test_build_default_name(self, mock_mkdir, mock_exists, mock_run, mock_which):
        mock_run.return_value.returncode = 0
        provider = SingularityImageProvider("/path/to/images")
        def_file = DefinitionFile(path="/path/to/container.def")
        result = provider.build(def_file, "/path/to/images")
        expected_path = "/path/to/images/container.sif"
        assert result == expected_path
        mock_run.assert_called_with(
            ["apptainer", "build", expected_path, "/path/to/container.def"],
            capture_output=True, text=True, cwd=str(Path("/path/to").resolve()),
        )

    @patch("shutil.which", side_effect=lambda x: "/usr/bin/" + x if x == "apptainer" else None)
    @patch("subprocess.run")
    @patch("pathlib.Path.exists", return_value=False)
    @patch("pathlib.Path.mkdir")
    def test_build_with_container_name(self, mock_mkdir, mock_exists, mock_run, mock_which):
        mock_run.return_value.returncode = 0
        provider = SingularityImageProvider("/path/to/images")
        def_file = DefinitionFile(
            path="/path/to/container.def", container_name="my-custom"
        )
        result = provider.build(def_file, "/path/to/images")
        expected_path = "/path/to/images/my-custom.sif"
        assert result == expected_path
        mock_run.assert_called_with(
            ["apptainer", "build", expected_path, "/path/to/container.def"],
            capture_output=True, text=True, cwd=str(Path("/path/to").resolve()),
        )

    @patch("shutil.which", side_effect=lambda x: "/usr/bin/" + x if x == "apptainer" else None)
    @patch("subprocess.run")
    @patch("pathlib.Path.exists", return_value=False)
    @patch("pathlib.Path.mkdir")
    def test_build_with_container_name_sif_extension(self, mock_mkdir, mock_exists, mock_run, mock_which):
        mock_run.return_value.returncode = 0
        provider = SingularityImageProvider("/path/to/images")
        def_file = DefinitionFile(
            path="/path/to/container.def", container_name="my-custom.sif"
        )
        result = provider.build(def_file, "/path/to/images")
        expected_path = "/path/to/images/my-custom.sif"
        assert result == expected_path
        mock_run.assert_called_with(
            ["apptainer", "build", expected_path, "/path/to/container.def"],
            capture_output=True, text=True, cwd=str(Path("/path/to").resolve()),
        )

    @patch("shutil.which", side_effect=[None, "/usr/bin/apptainer"])
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    def test_build_already_exists(self, mock_mkdir, mock_exists, mock_which):
        provider = SingularityImageProvider("/path/to/images")
        def_file = DefinitionFile(path="/path/to/container.def")
        result = provider.build(def_file, "/path/to/images")
        assert result == "/path/to/images/container.sif"

    @patch("shutil.which", side_effect=lambda x: "/usr/bin/" + x if x == "apptainer" else None)
    @patch("subprocess.run")
    @patch("pathlib.Path.exists", return_value=False)
    @patch("pathlib.Path.mkdir")
    def test_build_failure(self, mock_mkdir, mock_exists, mock_run, mock_which):
        mock_run.return_value.returncode = 1
        mock_run.return_value.stderr = "some error"
        provider = SingularityImageProvider("/path/to/images")
        def_file = DefinitionFile(path="/path/to/container.def")
        with pytest.raises(RuntimeError, match="Failed to build"):
            provider.build(def_file, "/path/to/images")

    @patch("shutil.which", return_value=None)
    @patch("pathlib.Path.mkdir")
    def test_build_no_executable(self, mock_mkdir, mock_which):
        provider = SingularityImageProvider("/path/to/images")
        def_file = DefinitionFile(path="/path/to/container.def")
        with pytest.raises(RuntimeError, match="No Apptainer/Singularity executable found"):
            provider.build(def_file, "/path/to/images")

    @patch("shutil.which", side_effect=lambda x: "/usr/bin/" + x if x == "apptainer" else None)
    @patch("subprocess.run")
    @patch("pathlib.Path.exists", return_value=False)
    @patch("pathlib.Path.mkdir")
    def test_build_with_build_flags(self, mock_mkdir, mock_exists, mock_run, mock_which):
        mock_run.return_value.returncode = 0
        provider = SingularityImageProvider("/path/to/images")
        def_file = DefinitionFile(
            path="/path/to/container.def",
            build_flags=("--ignore-fakeroot-command",),
        )
        result = provider.build(def_file, "/path/to/images")
        expected_path = "/path/to/images/container.sif"
        assert result == expected_path
        mock_run.assert_called_with(
            ["apptainer", "build", "--ignore-fakeroot-command",
             expected_path, "/path/to/container.def"],
            capture_output=True, text=True, cwd=str(Path("/path/to").resolve()),
        )


class MockImageProvider(ImageProvider):
    def __init__(self):
        self._executable = None

    def _get_executable(self) -> str:
        if self._executable is None:
            self._executable = "docker"
        return self._executable

    def exists(self, image: str) -> bool:
        return True

    def pull(self, image: str) -> bool:
        return True

    def build(self, def_file: DefinitionFile, output_dir: str = "") -> str:
        return "mock-built-image"

    def resolve_path(self, image: str) -> str:
        return image

    def prepare_image(self, image_def: ImageRef | ImageFile | DefinitionFile,
                      output_dir: str = "") -> str:
        if isinstance(image_def, ImageRef):
            return image_def.uri
        if isinstance(image_def, ImageFile):
            return str(image_def.path)
        if isinstance(image_def, DefinitionFile):
            return self.build(image_def, output_dir)
        raise Exception("Invalid image definition provided.")


class TestDockerContainerRuntime:
    def setup_method(self):
        self.provider = MockImageProvider()
        self.config = ContainerConfig(cuda_enabled=True, extra_flags=("--rm",))

    def test_build_run_command_with_cuda(self):
        rt = DockerContainerRuntime("ubuntu:22.04", self.provider, self.config)
        cmd = rt.build_run_command("ubuntu:22.04", ["echo", "hello"])
        assert cmd[0] == "docker"
        assert cmd[1] == "run"
        assert cmd[2] == "--gpus all"
        assert "--rm" in cmd
        img_idx = cmd.index("ubuntu:22.04")
        assert img_idx < len(cmd) - 2
        assert cmd[-2:] == ["echo", "hello"]

    def test_build_run_command_no_cuda(self):
        config = ContainerConfig(cuda_enabled=False)
        rt = DockerContainerRuntime("ubuntu:22.04", self.provider, config)
        cmd = rt.build_run_command("ubuntu:22.04", ["echo", "hello"])
        assert "--gpus all" not in cmd
        assert "--rm" in cmd 

    def test_build_exec_command(self):
        rt = DockerContainerRuntime("ubuntu:22.04", self.provider, self.config)
        cmd = rt.build_exec_command("ubuntu:22.04", ["bash", "-c", "ls"])
        assert cmd[0] == "docker"
        assert cmd[1] == "run"
        assert "--rm" in cmd
        assert "--gpus all" in cmd

    @patch("subprocess.run")
    def test_run_calls_subprocess(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        rt = DockerContainerRuntime("ubuntu:22.04", self.provider, self.config)
        rc = rt.run(["echo", "hello"])
        assert rc == 0

    @patch("subprocess.run")
    def test_execute_calls_subprocess(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        rt = DockerContainerRuntime("ubuntu:22.04", self.provider, self.config)
        rc = rt.execute(["bash", "-c", "ls"])
        assert rc == 0


class TestSingularityContainerRuntime:
    def setup_method(self):
        self.provider = MagicMock()
        self.provider._get_executable.return_value = "singularity"
        self.provider.resolve_path.return_value = "/path/to/images/myimage.sif"
        self.config = ContainerConfig(cuda_enabled=True, extra_flags=("--nv",))

    def test_build_run_command_with_cuda(self):
        rt = SingularityContainerRuntime("myimage.sif", self.provider, self.config)
        cmd = rt.build_run_command("/path/to/images/myimage.sif", ["echo", "hello"])
        assert cmd[0] == "singularity"
        assert cmd[1] == "run"
        assert "--nv" in cmd
        img_idx = cmd.index("/path/to/images/myimage.sif")
        assert img_idx < len(cmd) - 2

    def test_build_run_command_no_cuda(self):
        config = ContainerConfig(cuda_enabled=False)
        rt = SingularityContainerRuntime("myimage.sif", self.provider, config)
        cmd = rt.build_run_command("/path/to/images/myimage.sif", ["echo", "hello"])
        assert "--nv" not in cmd

    def test_build_exec_command(self):
        rt = SingularityContainerRuntime("myimage.sif", self.provider, self.config)
        cmd = rt.build_exec_command("/path/to/images/myimage.sif", ["bash"])
        assert cmd[0] == "singularity"
        assert cmd[1] == "exec"
        assert "--nv" in cmd

    @patch("subprocess.run")
    def test_run_calls_subprocess(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        rt = SingularityContainerRuntime("myimage.sif", self.provider, self.config)
        rc = rt.run(["echo", "hello"])
        assert rc == 0

    @patch("subprocess.run")
    def test_execute_calls_subprocess(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        rt = SingularityContainerRuntime("myimage.sif", self.provider, self.config)
        rc = rt.execute(["bash"])
        assert rc == 0



class TestBackendRegistry:
    def test_default_registrations(self):
        provider = MockImageProvider()
        config = ContainerConfig()
        rt = get_runtime_for_type(ContainerType.DOCKER, "ubuntu:22.04", provider, config)
        assert isinstance(rt, DockerContainerRuntime)

        rt = get_runtime_for_type(ContainerType.SINGULARITY, "myimage.sif", provider, config)
        assert isinstance(rt, SingularityContainerRuntime)

        rt = get_runtime_for_type(ContainerType.APPTAINER, "myimage.sif", provider, config)
        assert isinstance(rt, SingularityContainerRuntime)

    def test_unknown_type(self):
        import dsbo.container as cmod
        saved = cmod._container_runtime_registry.copy()
        cmod._container_runtime_registry.clear()
        provider = MockImageProvider()
        config = ContainerConfig()
        with pytest.raises(ValueError, match="Unknown container type"):
            get_runtime_for_type(ContainerType.DOCKER, "image", provider, config)
        cmod._container_runtime_registry.update(saved)


class TestContainerWrapperBase:
    @pytest.mark.parametrize(
        ("resources", "expected"),
        [
            (LocalWorkerSpec(cores=1, processes=1, memory="1GB"), False),
            (SLURMWorkerSpec(cores=1, processes=1, memory="1GB", walltime="00:10:00"), False),
            (SLURMWorkerSpec(cores=1, processes=1, memory="1GB", walltime="00:10:00", gpu_resources="gpu:1"), True),
        ],
    )
    def test_worker_has_gpu_from_resources(self, resources, expected):
        wrapper = ContainerWrapperBase(ImageRef(uri="ubuntu:22.04"))
        wrapper.resources = resources

        assert wrapper._worker_has_gpu() is expected

    @patch("dsbo.container.discover_container_environments", return_value={"docker": True})
    def test_cpu_worker_disables_cuda_for_runtime(self, mock_discover):
        wrapper = ContainerWrapperBase(ImageRef(uri="ubuntu:22.04"), cuda_enabled=True)
        wrapper.resources = LocalWorkerSpec(cores=1, processes=1, memory="1GB")
        provider = MockImageProvider()
        wrapper._make_provider = MagicMock(return_value=provider)

        cmd = wrapper._ContainerWrapperBase__construct_cmd(["echo"])

        assert "--gpus all" not in cmd

    @patch("torch.cuda.is_available", return_value=False)
    def test_worker_has_gpu_uses_local_torch_when_unassigned(self, mock_is_available):
        wrapper = ContainerWrapperBase(ImageRef(uri="ubuntu:22.04"))

        assert wrapper._worker_has_gpu() is False
        mock_is_available.assert_called_once_with()

    def test_explicit_container_type_override(self):
        wrapper = ContainerWrapperBase(
            ImageRef(uri="ubuntu:22.04"), container_type=ContainerType.SINGULARITY,
        )
        assert wrapper._detected_type == ContainerType.SINGULARITY

    def test_config_from_kwargs(self):
        wrapper = ContainerWrapperBase(
            ImageRef(uri="ubuntu:22.04"), container_extra_flags="--rm -v /foo:/bar",
            cuda_enabled=True,
        )
        assert wrapper._config.cuda_enabled is True
        assert wrapper._config.extra_flags == ("--rm", "-v", "/foo:/bar")

    def test_explicit_config(self):
        config = ContainerConfig(cuda_enabled=True, extra_flags=("--rm",))
        wrapper = ContainerWrapperBase(ImageRef(uri="ubuntu:22.04"), config=config)
        assert wrapper._config is config

    def test_container_extra_flags_property(self):
        wrapper = ContainerWrapperBase(
            ImageRef(uri="ubuntu:22.04"), container_extra_flags="--rm -v /foo:/bar",
        )
        assert wrapper.container_extra_flags == "--rm -v /foo:/bar"

    def test_container_extra_flags_property_none(self):
        wrapper = ContainerWrapperBase(ImageRef(uri="ubuntu:22.04"))
        assert wrapper.container_extra_flags is None


class TestImageRef:
    def test_normalize_uri_simple(self):
        ref = ImageRef(uri="ubuntu:22.04")
        assert ref._normalize_uri("ubuntu:22.04") == "ubuntu_22.04"

    def test_normalize_uri_with_slash(self):
        ref = ImageRef(uri="library/ubuntu:22.04")
        assert ref._normalize_uri("library/ubuntu:22.04") == "library_ubuntu_22.04"

    def test_normalize_uri_registry(self):
        ref = ImageRef(uri="myregistry.io/image:tag")
        assert ref._normalize_uri("myregistry.io/image:tag") == "myregistry.io_image_tag"

    def test_normalize_uri_strips_docker_scheme(self):
        ref = ImageRef(uri="docker://ubuntu:22.04")
        assert ref._normalize_uri("docker://ubuntu:22.04") == "ubuntu_22.04"

    def test_normalize_uri_strips_https_scheme(self):
        ref = ImageRef(uri="https://registry.io/image")
        assert ref._normalize_uri("https://registry.io/image") == "registry.io_image"

    def test_get_image_file_path_default(self):
        ref = ImageRef(uri="library/ubuntu:22.04")
        path = ref.get_image_file_path()
        expected = Path(os.getcwd(), "library_ubuntu_22.04.sif")
        assert path == expected

    def test_get_image_file_path_custom(self):
        ref = ImageRef(uri="library/ubuntu:22.04")
        path = ref.get_image_file_path("/path/to/images")
        assert path == Path("/path/to/images/library_ubuntu_22.04.sif")

    def test_get_image_file_path_custom_with_slashes(self):
        ref = ImageRef(uri="docker://ubuntu:22.04")
        path = ref.get_image_file_path("/path/to/images")
        assert path == Path("/path/to/images/ubuntu_22.04.sif")

    def test_frozen(self):
        ref = ImageRef(uri="ubuntu:22.04")
        with pytest.raises(AttributeError):
            ref.uri = "other"


class TestImageFile:
    def test_string_path(self):
        f = ImageFile(path="/path/to/images/myimage.sif")
        assert str(f.path) == "/path/to/images/myimage.sif"

    def test_path_object(self):
        f = ImageFile(path=Path("/path/to/images/myimage.sif"))
        assert isinstance(f.path, Path)

    def test_frozen(self):
        f = ImageFile(path="/tmp/test.sif")
        with pytest.raises(AttributeError):
            f.path = "/other.sif"



class TestDiscoverEnvironments:
    @patch("shutil.which", return_value="/usr/bin/docker")
    def test_docker_found(self, mock_which):
        envs = discover_container_environments()
        assert envs["docker"] is True

    @patch("shutil.which", side_effect=[None, "/usr/bin/apptainer", None])
    def test_apptainer_found(self, mock_which):
        envs = discover_container_environments()
        assert envs["docker"] is False
        assert envs["apptainer"] is True
        assert envs["singularity"] is False

    @patch("shutil.which", return_value=None)
    def test_none_found(self, mock_which):
        envs = discover_container_environments()
        assert all(v is False for v in envs.values())


class TestImageTypeRequired:
    def test_image_file_requires_singularity(self):
        assert image_type_required(ImageFile(path="foo.sif")) == ImageRequirement.SINGULARITY_ONLY

    def test_image_ref_allows_either(self):
        assert image_type_required(ImageRef(uri="ubuntu:22.04")) == ImageRequirement.EITHER


class TestSelectCompatibleEnvironment:
    def test_singularity_only_with_docker_skipped(self):
        available = {"docker": True, "apptainer": False, "singularity": False}
        with pytest.raises(Exception, match="reuires"):
            select_compatible_environment(ImageRequirement.SINGULARITY_ONLY, available)

    def test_singularity_only_with_apptainer(self):
        available = {"docker": True, "apptainer": True, "singularity": False}
        assert select_compatible_environment(ImageRequirement.SINGULARITY_ONLY, available) == ContainerType.APPTAINER

    def test_singularity_only_with_singularity(self):
        available = {"docker": False, "apptainer": False, "singularity": True}
        assert select_compatible_environment(ImageRequirement.SINGULARITY_ONLY, available) == ContainerType.SINGULARITY

    def test_either_with_docker_only(self):
        available = {"docker": True, "apptainer": False, "singularity": False}
        assert select_compatible_environment(ImageRequirement.EITHER, available) == ContainerType.DOCKER

    def test_either_prefers_apptainer(self):
        available = {"docker": True, "apptainer": True, "singularity": True}
        assert select_compatible_environment(ImageRequirement.EITHER, available) == ContainerType.APPTAINER

    def test_either_prefers_singularity_over_docker(self):
        available = {"docker": True, "apptainer": False, "singularity": True}
        assert select_compatible_environment(ImageRequirement.EITHER, available) == ContainerType.SINGULARITY

    def test_none_available(self):
        available = {"docker": False, "apptainer": False, "singularity": False}
        with pytest.raises(Exception, match="reuires"):
            select_compatible_environment(ImageRequirement.EITHER, available)


class TestCheckMismatch:
    def test_image_file_with_docker_raises(self):
        with pytest.raises(RuntimeError, match="ImageFile \\(.sif\\) used with Docker"):
            check_mismatch(ImageFile(path="foo.sif"), ContainerType.DOCKER)

    def test_image_file_with_singularity_ok(self):
        check_mismatch(ImageFile(path="foo.sif"), ContainerType.SINGULARITY)

    def test_image_ref_with_singularity_ok(self):
        check_mismatch(ImageRef(uri="ubuntu:22.04"), ContainerType.SINGULARITY)

    def test_image_ref_with_docker_ok(self):
        check_mismatch(ImageRef(uri="ubuntu:22.04"), ContainerType.DOCKER)






class TestPullImageRefToSif:
    @patch("shutil.which", side_effect=["/usr/bin/apptainer", None])
    @patch("subprocess.run")
    @patch("pathlib.Path.exists", return_value=False)
    @patch("pathlib.Path.mkdir")
    def test_pull_apptainer(self, mock_mkdir, mock_exists, mock_run, mock_which):
        mock_run.return_value.returncode = 0
        ref = ImageRef(uri="docker://ubuntu:22.04")
        provider = SingularityImageProvider(image_dir="/path/to/images")
        result = provider._pull_ref_to_sif(ref, "/path/to/images")
        expected = "/path/to/images/ubuntu_22.04.sif"
        assert result == expected
        mock_run.assert_called_with(
            ["singularity", "pull", expected, "docker://ubuntu:22.04"],
            capture_output=True, text=True,
        )

    @patch("shutil.which", side_effect=[None, "/usr/bin/singularity"])
    @patch("subprocess.run")
    @patch("pathlib.Path.exists", return_value=False)
    @patch("pathlib.Path.mkdir")
    def test_pull_singularity(self, mock_mkdir, mock_exists, mock_run, mock_which):
        mock_run.return_value.returncode = 0
        ref = ImageRef(uri="docker://ubuntu:22.04")
        provider = SingularityImageProvider(image_dir="/path/to/images")
        result = provider._pull_ref_to_sif(ref, "/path/to/images")
        expected = "/path/to/images/ubuntu_22.04.sif"
        assert result == expected
        mock_run.assert_called_with(
            ["apptainer", "pull", expected, "docker://ubuntu:22.04"],
            capture_output=True, text=True,
        )

    @patch("shutil.which", return_value=None)
    @patch("pathlib.Path.mkdir")
    def test_no_executable(self, mock_mkdir, mock_which):
        ref = ImageRef(uri="docker://ubuntu:22.04")
        provider = SingularityImageProvider(image_dir="/path/to/images")
        with pytest.raises(RuntimeError, match="No Singularity/Apptainer executable found"):
            provider._pull_ref_to_sif(ref, "/path/to/images")

    @patch("shutil.which", side_effect=["/usr/bin/apptainer", None])
    @patch("pathlib.Path.exists", return_value=True)
    @patch("pathlib.Path.mkdir")
    def test_already_exists(self, mock_mkdir, mock_exists, mock_which):
        ref = ImageRef(uri="docker://ubuntu:22.04")
        provider = SingularityImageProvider(image_dir="/path/to/images")
        result = provider._pull_ref_to_sif(ref, "/path/to/images")
        assert result == "/path/to/images/ubuntu_22.04.sif"

    @patch("shutil.which", side_effect=["/usr/bin/apptainer", None])
    @patch("subprocess.run")
    @patch("pathlib.Path.exists", return_value=False)
    @patch("pathlib.Path.mkdir")
    def test_pull_failure(self, mock_mkdir, mock_exists, mock_run, mock_which):
        mock_run.return_value = MagicMock(returncode=1)
        ref = ImageRef(uri="docker://ubuntu:22.04")
        provider = SingularityImageProvider(image_dir="/path/to/images")
        with pytest.raises(RuntimeError, match="Failed to pull"):
            provider._pull_ref_to_sif(ref, "/path/to/images")

    @patch("shutil.which", side_effect=["/usr/bin/apptainer", None])
    @patch("subprocess.run")
    @patch("pathlib.Path.exists", return_value=False)
    @patch("pathlib.Path.mkdir")
    def test_pull_plain_uri_adds_docker_scheme(self, mock_mkdir, mock_exists, mock_run, mock_which):
        """Plain 'ubuntu:22.04' should get docker:// prefix for singularity."""
        mock_run.return_value.returncode = 0
        ref = ImageRef(uri="ubuntu:22.04")
        provider = SingularityImageProvider(image_dir="/path/to/images")
        result = provider._pull_ref_to_sif(ref, "/path/to/images")
        expected = "/path/to/images/ubuntu_22.04.sif"
        assert result == expected
        mock_run.assert_called_with(
            ["singularity", "pull", expected, "docker://ubuntu:22.04"],
            capture_output=True, text=True,
        )

    @patch("shutil.which", side_effect=[None, "/usr/bin/singularity"])
    @patch("subprocess.run")
    @patch("pathlib.Path.exists", return_value=False)
    @patch("pathlib.Path.mkdir")
    def test_pull_plain_uri_with_apptainer(self, mock_mkdir, mock_exists, mock_run, mock_which):
        """Plain URI with apptainer executable."""
        mock_run.return_value.returncode = 0
        ref = ImageRef(uri="ubuntu:22.04")
        provider = SingularityImageProvider(image_dir="/path/to/images")
        result = provider._pull_ref_to_sif(ref, "/path/to/images")
        expected = "/path/to/images/ubuntu_22.04.sif"
        assert result == expected
        mock_run.assert_called_with(
            ["apptainer", "pull", expected, "docker://ubuntu:22.04"],
            capture_output=True, text=True,
        )

    def test_ensure_docker_scheme_preserves_docker_prefix(self):
        provider = SingularityImageProvider(image_dir="/path/to/images")
        assert provider._ensure_docker_scheme("docker://ubuntu:22.04") == "docker://ubuntu:22.04"

    def test_ensure_docker_scheme_preserves_https_prefix(self):
        provider = SingularityImageProvider(image_dir="/path/to/images")
        assert provider._ensure_docker_scheme("https://registry.io/image") == "https://registry.io/image"

    def test_ensure_docker_scheme_adds_prefix(self):
        provider = SingularityImageProvider(image_dir="/path/to/images")
        assert provider._ensure_docker_scheme("ubuntu:22.04") == "docker://ubuntu:22.04"
        assert provider._ensure_docker_scheme("library/ubuntu:22.04") == "docker://library/ubuntu:22.04"


    @patch("shutil.which", return_value="/usr/bin/docker")
    def test_docker_prepare_image_with_plain_uri(self, mock_which):
        """Docker should pass plain URIs through as-is."""
        provider = DockerImageProvider()
        ref = ImageRef(uri="ubuntu:22.04")
        result = provider.prepare_image(ref)
        assert result == "ubuntu:22.04"

    @patch("shutil.which", return_value="/usr/bin/docker")
    def test_docker_prepare_image_with_docker_scheme_uri(self, mock_which):
        """Docker should pass docker:// URIs through as-is (Docker handles both)."""
        provider = DockerImageProvider()
        ref = ImageRef(uri="docker://ubuntu:22.04")
        result = provider.prepare_image(ref)
        assert result == "docker://ubuntu:22.04"


class TestImageDefinitionFile:
    def test_string_path(self):
        f = DefinitionFile(path="/path/to/Dockerfile")
        assert str(f.path) == "/path/to/Dockerfile"

    def test_path_object(self):
        f = DefinitionFile(path=Path("/path/to/Dockerfile"))
        assert isinstance(f.path, Path)

    def test_frozen(self):
        f = DefinitionFile(path="/path/to/Dockerfile")
        with pytest.raises(AttributeError):
            f.path = "/other"

    def test_def_type_dockerfile(self):
        f = DefinitionFile(path="/path/to/Dockerfile")
        assert f._def_type() == "docker"

    def test_def_type_dockerfile_lowercase(self):
        f = DefinitionFile(path="/path/to/dockerfile")
        assert f._def_type() == "docker"

    def test_def_type_dot_def(self):
        f = DefinitionFile(path="/path/to/container.def")
        assert f._def_type() == "singularity"

    def test_def_type_unknown_defaults_to_docker(self):
        f = DefinitionFile(path="/path/to/container.txt")
        assert f._def_type() == "docker"

    def test_container_name_defaults_to_none(self):
        f = DefinitionFile(path="/path/to/Dockerfile")
        assert f.container_name is None

    def test_container_name_custom(self):
        f = DefinitionFile(path="/path/to/Dockerfile", container_name="my-custom-image")
        assert f.container_name == "my-custom-image"

    def test_build_flags_defaults_to_empty(self):
        f = DefinitionFile(path="/path/to/Dockerfile")
        assert f.build_flags == ()

    def test_build_flags_custom(self):
        f = DefinitionFile(
            path="/path/to/Singularity.def",
            build_flags=("--ignore-fakeroot-command", "--force"),
        )
        assert f.build_flags == ("--ignore-fakeroot-command", "--force")


class TestNormalizePathTag:
    def test_simple_stem(self):
        assert _normalize_path_tag("/path/to/Dockerfile") == "dockerfile"

    def test_stem_with_special_chars(self):
        assert _normalize_path_tag("/path/to/my-def.def") == "my-def"

    def test_stem_with_colon(self):
        assert _normalize_path_tag("/path/to/image:v1.def") == "image_v1"



class TestImageTypeRequiredWithDef:
    def test_dockerfile_requires_docker(self):
        assert image_type_required(
            DefinitionFile(path="/path/to/Dockerfile")
        ) == ImageRequirement.DOCKER_ONLY

    def test_dot_def_requires_singularity(self):
        assert image_type_required(
            DefinitionFile(path="/path/to/container.def")
        ) == ImageRequirement.SINGULARITY_ONLY


class TestSelectCompatibleEnvironmentWithDef:
    def test_docker_only_with_apptainer_skipped(self):
        available = {"docker": False, "apptainer": True, "singularity": False}
        with pytest.raises(Exception, match="reuires"):
            select_compatible_environment(ImageRequirement.DOCKER_ONLY, available)

    def test_docker_only_with_docker(self):
        available = {"docker": True, "apptainer": False, "singularity": False}
        assert select_compatible_environment(ImageRequirement.DOCKER_ONLY, available) == ContainerType.DOCKER

    def test_docker_only_prefers_docker(self):
        available = {"docker": True, "apptainer": True, "singularity": True}
        assert select_compatible_environment(ImageRequirement.DOCKER_ONLY, available) == ContainerType.DOCKER


class TestCheckMismatchWithDef:
    def test_dockerfile_with_apptainer_raises(self):
        with pytest.raises(RuntimeError, match="Dockerfiles cannot be built"):
            check_mismatch(DefinitionFile(path="/path/to/Dockerfile"), ContainerType.APPTAINER)

    def test_dockerfile_with_singularity_raises(self):
        with pytest.raises(RuntimeError, match="Dockerfiles cannot be built"):
            check_mismatch(DefinitionFile(path="/path/to/Dockerfile"), ContainerType.SINGULARITY)

    def test_dockerfile_with_docker_ok(self):
        check_mismatch(DefinitionFile(path="/path/to/Dockerfile"), ContainerType.DOCKER)

    def test_dot_def_with_docker_raises(self):
        with pytest.raises(RuntimeError, match="Singularity definition files cannot be built"):
            check_mismatch(DefinitionFile(path="/path/to/container.def"), ContainerType.DOCKER)

    def test_dot_def_with_singularity_ok(self):
        check_mismatch(DefinitionFile(path="/path/to/container.def"), ContainerType.SINGULARITY)





class TestSubprocessRun:
    @patch("subprocess.Popen")
    def test_streams_callbacks(self, mock_popen):
        mock_process = MagicMock()
        mock_process.stdout.readline = MagicMock(side_effect=["line1\n", "line2\n", ""])
        mock_process.stderr.readline = MagicMock(side_effect=["err1\n", ""])
        mock_process.wait.return_value = 0
        mock_popen.return_value = mock_process

        stdout_lines = []
        stderr_lines = []

        rc = subprocess_run(
            ["echo", "hello"],
            lambda x: stdout_lines.append(x),
            lambda x: stderr_lines.append(x),
        )

        assert rc == 0
        assert stdout_lines == ["line1\n", "line2\n"]
        assert stderr_lines == ["err1\n"]


def make_container_module(ref):
    class ContainerModule(Module, ContainerWrapperBase):
        def __init__(self):
            Module.__init__(self)
            ContainerWrapperBase.__init__(self, ref)
    return ContainerModule()


class TestWarmContainerDependencies:
    def test_empty_workflow_returns_empty_dict(self):
        wf = Workflow(Module())
        result = warm_container_dependencies(wf, verbose=False)
        assert result == {}

    def test_deduplicates_identical_image_defs(self):
        ref = ImageRef(uri="ubuntu:22.04")
        a = make_container_module(ref)
        b = make_container_module(ref)
        wf = Workflow(DependencyDict({a: [], b: []}))

        with patch("dsbo.container.discover_container_environments",
                   return_value={"docker": True, "apptainer": False, "singularity": False}), \
             patch.object(DockerImageProvider, "prepare_image", return_value="ubuntu:22.04") as mock_prep:
            result = warm_container_dependencies(wf, verbose=False)

        mock_prep.assert_called_once()
        assert result == {ref: "ubuntu:22.04"}

    def test_raises_on_prepare_image_failure(self):
        ref = ImageRef(uri="nonexistent:tag")
        a = make_container_module(ref)
        wf = Workflow(a)

        with patch("dsbo.container.discover_container_environments",
                   return_value={"docker": True, "apptainer": False, "singularity": False}), \
             patch.object(DockerImageProvider, "prepare_image",
                          side_effect=RuntimeError("pull failed")):
            with pytest.raises(RuntimeError, match="pull failed"):
                warm_container_dependencies(wf, verbose=False)

    def test_warms_mixed_image_types(self):
        img_ref = ImageRef(uri="ubuntu:22.04")
        img_file = ImageFile(path="/path/to/existing.sif")
        def_file = DefinitionFile(path="/path/to/Dockerfile")

        a = make_container_module(img_ref)
        b = make_container_module(img_file)
        c = make_container_module(def_file)
        wf = Workflow(DependencyDict({a: [], b: [], c: []}))

        with patch("dsbo.container.discover_container_environments",
                   return_value={"docker": True, "apptainer": True, "singularity": False}), \
             patch.object(DockerImageProvider, "prepare_image", return_value="ubuntu:22.04"), \
             patch.object(SingularityImageProvider, "prepare_image") as mock_sing_prep:
            mock_sing_prep.side_effect = lambda idef, od: (
                str(idef.path) if isinstance(idef, ImageFile) else "built-image"
            )
            result = warm_container_dependencies(wf, verbose=False)

        assert len(result) == 3
        assert img_ref in result
        assert img_file in result
        assert def_file in result

    def test_shows_progress_bar_when_verbose(self, capsys):
        ref_a = ImageRef(uri="ubuntu:22.04")
        ref_b = ImageRef(uri="python:3.11")

        a = make_container_module(ref_a)
        b = make_container_module(ref_b)
        wf = Workflow(DependencyDict({a: [], b: []}))

        with patch("dsbo.container.discover_container_environments",
                   return_value={"docker": True, "apptainer": False, "singularity": False}), \
             patch.object(DockerImageProvider, "prepare_image",
                          side_effect=lambda idef, od: idef.uri):
            result = warm_container_dependencies(wf, verbose=True)

        captured = capsys.readouterr()
        assert "Warming containers" in captured.err
        assert len(result) == 2

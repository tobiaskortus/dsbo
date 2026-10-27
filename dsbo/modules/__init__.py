from __future__ import annotations

import torch
import inspect
import weakref
from enum import Enum
from pathlib import Path
from abc import ABC, abstractmethod
from functools import partial, wraps
from typing import BinaryIO, Optional, Callable, TYPE_CHECKING


from dsbo.autograd import autograd_context
from dsbo.logger import WorkflowTaskLogger
from dsbo.modules.dsl import ModuleDslMixin
from contextlib import nullcontext, suppress
from dsbo.utils.compile import dryrunnable, is_dry_run
from dsbo.utils.io import (
    AbstractFilePath,
    Input,
    Output,
    Workdir,
    get_workdir,
    with_remote_workdir,
)
from dsbo.utils.io.workdir import DefaultPaths

if TYPE_CHECKING:
    from dsbo.storage.artifact import ArtifactRef
    from dsbo.worker.spec import WorkerSpec
    from dsbo.surrogates.module_surrogate import ModuleSurrogate
    from dsbo.surrogates.module_surrogate import ModuleSurrogateContract


def ensure_dict_return(fn):
    @wraps(fn)
    def _inner(*args, **kwargs):
        res = fn(*args, **kwargs)
        return {} if res is None else res
    return _inner


def auto_wrap(cls, method_name, *decorators):
    raw = cls.__dict__.get(method_name)
    if raw is None or getattr(raw, "_dsbo_wrapped", False):
        return
    wrapped = raw
    for decorator in decorators:
        wrapped = decorator(wrapped)
    setattr(wrapped, "_dsbo_wrapped", True)
    setattr(cls, method_name, wrapped)


class Module(ModuleDslMixin, ABC):
    requires_design_tensor = False
    supports_forward_ad = True

    def __init__(self):
        self._registered_files: dict[str, AbstractFilePath] = {}
        self._registered_inputs: list[AbstractFilePath] = []
        self._registered_outputs: list[AbstractFilePath] = []

        self._alias: str | None = None
        self._seed: int | None = None
        self._no_grad_op: bool = False

        self.i: int | None = None
        self.resources: WorkerSpec | None = None

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        auto_wrap(cls, "__call__", dryrunnable, ensure_dict_return)
        auto_wrap(cls, "backward", with_remote_workdir)

    @dryrunnable
    def __call__(self, design: dict, **kwargs) -> Optional[dict[str, float]]:
        pass

    def __setattr__(self, name, value):
        if not hasattr(self, "_registered_files") and name != "_registered_files":
            raise RuntimeError(
                "Subclasses of Module must call super().__init__() in their __init__ before setting attributes."
            )

        super().__setattr__(name, value)

        #TODO: define a clear policy when Inputs and Outputs should be registered
        #TODO: and when AbstractFilePath should be allowed
        if isinstance(value, AbstractFilePath):
            self._registered_files[name] = value

        if isinstance(value, (Input, Output)):
            self._registered_files[name] = value.path

            if isinstance(value, Input):
                self._registered_inputs += [value.path]
            else:
                self._registered_outputs += [value.path]


    def __repr__(self):
        return self.name
    
    @property
    def surrogate_contract(self) -> ModuleSurrogateContract:
        raise NotImplementedError("No surrogate contract defined for this module.")

    def patch(self, surrogate: ModuleSurrogate):
        surrogate.register_contract(self.surrogate_contract)
        self.surrogate = surrogate
        surrogate.model_file_path = Output(
            Workdir.register_file(
                DefaultPaths.SURROGATE,
                f"{surrogate._artifact_name}_{self.name}.pt",
            )
        )
        return self

    @property
    def name(self) -> str:
        suffix = f"[{self.i}]" if self.i is not None else ""
        name = type(self).__name__ if self._alias is None else self._alias
        return f"{name}{suffix}"

    
    @autograd_context
    @with_remote_workdir
    def run(self, design: dict, **kwargs):
        with nullcontext() if (is_dry_run() or get_workdir() is None) else WorkflowTaskLogger(self):
            with Workdir.use_file_suffix(str(self.i)) if self.i is not None else nullcontext():
                return self(design=design, **kwargs)

    @with_remote_workdir
    def run_backward(self, design: dict, grad_output: dict | None = None, design_tensor: torch.Tensor | None = None) -> None:
        with Workdir.use_file_suffix(str(self.i)) if self.i is not None else nullcontext():
            if design_tensor is None:
                self.backward(design, grad_output)
            else:
                self.backward(design, grad_output, design_tensor=design_tensor)

    def backward(self, design: dict, grad_output: dict | None = None, **kwargs) -> None:
        pass

    def to(self, resource: WorkerSpec) -> Module:
        self.resources = resource
        return self
    
    def set_alias(self, alias: str) -> Module:
        self._alias = alias
        return self

    def set_seed(self, seed: int) -> Module:
        self._seed = seed
        return self
    
    def op_propagates_grad_(self, propagate_grads: bool = True) -> Module:
        self._no_grad_op = not propagate_grads
        return self

    @property
    def registered_files(self) -> dict[str, AbstractFilePath]:
        return self._registered_files

    @property
    def registered_inputs(self) -> list[AbstractFilePath]:
        return list(self._registered_inputs)

    @property
    def registered_outputs(self) -> list[AbstractFilePath]:
        return list(self._registered_outputs)


def _restore_dynamic_submodule(parent, method_name, bindings, index, resources):
    submodule = getattr(parent, method_name)
    if bindings:
        submodule.bind(**bindings)
    submodule.i = index
    submodule.resources = resources
    return submodule


def dynamic_submodule(wrapped_fn):
    class DynamicSubmodule:
        def __init__(self, wrapped_fn: Callable):
            self._wrapped_fn = wrapped_fn
            self._wrapped_backward = None
            self._submodule_cache = weakref.WeakKeyDictionary()


        def backward(self,func: Callable):
            self._assert_valid_backward_signature(func)
            self._assert_backward_is_none()

            self._wrapped_backward = func
            

        def _assert_backward_is_none(self):
            assert self._wrapped_backward == None, \
                    f"The backward function for {self._wrapped_fn.__name__} is " +\
                    f"already defined as {self._wrapped_backward.__name__}"

        def _assert_valid_backward_signature(self, func: Callable):
            args = inspect.getfullargspec(func).args
            with suppress(ValueError, AttributeError):
                args.remove("self")
                args.remove("args")
                args.remove("kwargs")

            assert args == ["design", "grad_output"], \
                    f"Invalid backward arguments. Got {args} but backwards " +\
                    f"requires [design, grad_output]"

        def __getstate__(self):
            return {"_wrapped_fn": self._wrapped_fn, "_wrapped_backward": self._wrapped_backward}

        def __setstate__(self, state):
            self._wrapped_fn = state["_wrapped_fn"]
            self._wrapped_backward = state["_wrapped_backward"]
            self._submodule_cache = weakref.WeakKeyDictionary()

        def __get__(self, obj, *args, **kwargs):
            if obj is None:
                return self._wrapped_fn
            
            if not isinstance(obj, Module):
                raise TypeError(
                    f"@dynamic_submodule must be accessed on a `Module` instance; got {type(obj).__name__}."
                )

            class Submodule(Module):
                def __init__(self, parent: Module, func: Callable, backward: Callable):
                    object.__setattr__(self, "_parent", parent)
                    object.__setattr__(self, "_func", func)        
                    object.__setattr__(self, "_backward", backward)
                    object.__setattr__(self, "_bound", partial(func, parent))

                    if backward is not None:
                        def backward_fn(design, grad_outputs=None):
                            backward(self, design, grad_outputs)

                        object.__setattr__(self, "backward", backward_fn)

                def __setattr__(self, name, value):
                    if name in ("_parent", "_func", "_backward", "fn", "i", "resources") or name.startswith("_"):
                        object.__setattr__(self, name, value)
                    else:
                        setattr(self._parent, name, value)

                def bind(self, **kwargs):
                    self._bound = partial(self._func, self._parent, **kwargs)
                    return self

                def binding(self, name):
                    return self._bound.keywords[name]

                def __call__(self, design, **kwargs):
                    try:
                        return self._bound(design, **kwargs)
                    except Exception as e:
                        tb = e.__traceback__
                        if tb is not None:
                            tb = tb.tb_next
                        raise e.with_traceback(tb)

                def __getattr__(self, name):
                    return getattr(self._parent, name)

                def __reduce__(self):
                    return (
                        _restore_dynamic_submodule,
                        (
                            self._parent,
                            self._func.__name__,
                            self._bound.keywords,
                            getattr(self, "i", None),
                            getattr(self, "resources", None),
                        ),
                    )
            
                
                @property
                def name(self) -> str:
                    suffix = f"[{self.i}]" if self.i is not None else ""
                    return f"{type(self._parent).__name__}.{self._func.__name__}{suffix}"

                @property
                def registered_inputs(self) -> list[AbstractFilePath]:
                    return self._parent.registered_inputs

                @property
                def registered_outputs(self) -> list[AbstractFilePath]:
                    return self._parent.registered_outputs

                @property
                def registered_files(self) -> dict[str, AbstractFilePath]:
                    return self._parent.registered_files

            submodule = self._submodule_cache.get(obj)

            if submodule is not None:
                return submodule
        
            submodule = Submodule(obj, self._wrapped_fn, self._wrapped_backward)
            self._submodule_cache[obj] = submodule

            return submodule
        

    return DynamicSubmodule(wrapped_fn)


class ModelLifecycleStage(Enum):
    TRAIN = ("train", "trained")
    PRETRAIN = ("pretrain", "pretrained")
    FINETUNE = ("finetune", "finetuned")

    @property
    def task_name(self) -> str:
        return self.value[0]

    @property
    def artifact_stage(self) -> str:
        return self.value[1]


class TrainableModule(Module):
    def __init__(self, model_dir: str = "models"):
        super().__init__()
        self.model_file_path = Output(
            Workdir.register_file(model_dir, f"{self._artifact_name}.pt")
        )
        self._lifecycle_tasks: dict[ModelLifecycleStage, Module] = {}
        self._task_stages: dict[Module, ModelLifecycleStage] = {}

        for stage in ModelLifecycleStage:
            if self.implements(stage.task_name):
                task = getattr(self, stage.task_name)
                if isinstance(task, Module):
                    self._lifecycle_tasks[stage] = task
                    self._task_stages[task] = stage

        if not self.implements_train and not self.implements_pretrain:
            raise NotImplementedError(
                f"{type(self).__name__} must implement train or pretrain "
                "to be trained through the workflow."
            )

    def implements(self, method_name: str) -> bool:
        for cls in type(self).__mro__:
            if method_name in cls.__dict__:
                return cls is not TrainableModule
        return False

    @dynamic_submodule
    def train(self, design, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} does not implement train.")

    @dynamic_submodule
    def pretrain(self, design, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} does not implement pretrain.")

    @dynamic_submodule
    def finetune(self, design, **kwargs):
        raise NotImplementedError(f"{type(self).__name__} does not implement finetune.")

    @abstractmethod
    def state_dict(self) -> dict:
        ...

    @abstractmethod
    def from_state_dict(self, state: dict) -> None:
        ...

    def save_to_file(self, model_file: str | Path) -> None:
        torch.save(self.state_dict(), model_file)

    def load_from_file(self, model_file: str | Path | BinaryIO) -> None:
        self.from_state_dict(
            torch.load(model_file, weights_only=True, map_location="cpu")
        )

    def publish_trained_model(self, model_file: str, stage: str = "trained") -> ArtifactRef:
        from dsbo.storage import get_current_storage

        storage = get_current_storage()
        artifact = storage.artifacts.create_artifact(self._artifact_name, "model")
        artifact.add_file(model_file, name="model.pt")
        return storage.artifacts.save_artifact(artifact, stage=stage).ref

    def training_stage(self, task: Module) -> str:
        stage = self._task_stages.get(task)
        if stage is None:
            raise ValueError(f"{task.name} is not a lifecycle task of {type(self).__name__}.")
        return stage.artifact_stage

    def select_training_task(self) -> Module:
        return self._lifecycle_tasks[self._get_next_stage()]

    def _resolve_model_ref(self) -> ArtifactRef:
        from dsbo.storage import get_current_storage
        from dsbo.storage.artifact import ArtifactRef

        storage = get_current_storage()
        name = self._artifact_name
        for stage in (ModelLifecycleStage.FINETUNE, ModelLifecycleStage.PRETRAIN, ModelLifecycleStage.TRAIN):
            artifact_stage = stage.artifact_stage
            version = storage.artifacts.latest_version(name, stage=artifact_stage)
            if version is not None:
                return ArtifactRef(name, "model", version, artifact_stage)
        raise RuntimeError(f"No trained model artifact exists for '{name}'.")

    def _get_next_stage(self) -> ModelLifecycleStage:
        if self.implements_finetune and self.is_pretrained:
            return ModelLifecycleStage.FINETUNE
        if self.implements_pretrain:
            return ModelLifecycleStage.PRETRAIN
        return ModelLifecycleStage.TRAIN

    def _has_artifact_for(self, *stages: ModelLifecycleStage) -> bool:
        from dsbo.storage import get_current_storage

        if not (storage := get_current_storage()):
            return False
        return any(
            storage.artifacts.latest_version(self._artifact_name, stage=stage.artifact_stage) is not None
            for stage in stages
        )

    @property
    def implements_train(self) -> bool:
        return ModelLifecycleStage.TRAIN in self._lifecycle_tasks

    @property
    def implements_pretrain(self) -> bool:
        return ModelLifecycleStage.PRETRAIN in self._lifecycle_tasks

    @property
    def implements_finetune(self) -> bool:
        return ModelLifecycleStage.FINETUNE in self._lifecycle_tasks

    @property
    def implements_pretrain_and_finetune(self) -> bool:
        return self.implements_pretrain and self.implements_finetune

    @property
    def is_available(self) -> bool:
        return self.is_pretrained or self.is_finetuned

    @property
    def is_pretrained(self) -> bool:
        return self._has_artifact_for(ModelLifecycleStage.PRETRAIN, ModelLifecycleStage.TRAIN)

    @property
    def is_finetuned(self) -> bool:
        return self._has_artifact_for(ModelLifecycleStage.FINETUNE)

    @property
    def _artifact_name(self) -> str:
        return type(self).__name__.lower()


from dsbo.modules.dsl import with_no_grad
from dsbo.modules.fused import BatchFusedModule, FusedModule, batch_fuse

import time
import pytest
from unittest.mock import patch

from dsbo.modules import Module, dynamic_submodule
from dsbo.utils.io import AbstractFilePath, DefaultPaths, Input, Output, Workdir

class EmptyModuleMock(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwds):
        pass

class DefaultModuleMock(Module):
    def __init__(self):
        super().__init__()

    def __call__(self, *args, **kwds):
        time.sleep(0.01)
        return {"result": 42}


def test_dynamic_submodule_inherits_parent_registered_files():
    class Parent(Module):
        def __init__(self):
            super().__init__()
            self.train_input = Input(Workdir.register_file("simulation", "train.h5"))
            self.val_input = Input(Workdir.register_file("simulation", "val.h5"))
            self.model = Output(Workdir.register_file("reconstruction", "model.pt"))

        @dynamic_submodule
        def train(self, design):
            pass

    parent = Parent()
    assert parent.train.registered_inputs == parent.registered_inputs
    assert parent.train.registered_outputs == parent.registered_outputs


@patch('dsbo.modules.dryrunnable', return_value=False)
def test_module_init_subclass_wrapped(dryrunnable_mock):
    class ModuleModuleModule(Module):
        def __call__(self, *a, **kw): pass


    module = ModuleModuleModule()
    assert module.__call__._dsbo_wrapped is True
    assert dryrunnable_mock.call_count == 1


@patch('dsbo.utils.compile.is_dry_run', return_value=True)
@patch('time.sleep')
def test_dryrunnable(mock_sleep, mock_dry_run):
    module = DefaultModuleMock()
    result = module()

    #Make sure that the original function is not executed
    assert mock_sleep.call_count == 0


def test_ensure_dict_return():
    module = EmptyModuleMock()
    result = module()

    assert isinstance(result, dict)
    assert len(result) == 0


def test_register_files():
    module = DefaultModuleMock()
    module.file1 = AbstractFilePath(path=DefaultPaths.SIMULATION, filename="file1.txt")
    module.file2 = AbstractFilePath(path=DefaultPaths.SIMULATION, filename="file1.txt")

    assert "file1" in module.registered_files
    assert "file2" in module.registered_files


def test_exception_on_uninitialized_subclass():
    with pytest.raises(RuntimeError) as exc_info:
        class BadModule(Module):
            def __init__(self):
                self.file = AbstractFilePath(path=DefaultPaths.SIMULATION, filename="file.txt")

        module = BadModule()

    assert "Subclasses of Module must call super().__init__()" in str(exc_info.value)

def test_module_name_and_repr():
    module = DefaultModuleMock()
    assert module.name == "DefaultModuleMock"
    assert repr(module) == "DefaultModuleMock"


def test_module_name_and_repr_with_index():
    module = DefaultModuleMock()
    module.i = 1
    assert module.name == "DefaultModuleMock[1]"
    assert repr(module) == "DefaultModuleMock[1]"


def test_dynamic_submodule_as_module():
    class DynamicSubmodule(Module):
        def __init__(self):
            super().__init__()

        @dynamic_submodule
        def train():
            pass

    submodule = DynamicSubmodule()
    assert isinstance(submodule.train, Module)


def test_dynamic_submodule_as_method():
    class DynamicSubmodule(Module):
        def __init__(self):
            super().__init__()

        @dynamic_submodule
        def train(self, *args, **kwargs):
            return {"result": 42}

    submodule = DynamicSubmodule()
    result = submodule.train({})
    assert result == {"result": 42}


def test_dynamic_submodule_exception_stacktrace():
    class DynamicSubmodule(Module):
        def __init__(self):
            super().__init__()

        @dynamic_submodule
        def train(self, *args, **kwargs):
            raise ValueError()

    module = DynamicSubmodule()
    submodule = module.train
    with pytest.raises(ValueError) as exc_info:
        submodule({})

    # Ensure the stacktrace does not expose the internal Submodule wrapper class
    import traceback as _tb
    tb_text = "".join(_tb.format_tb(exc_info.value.__traceback__))
    assert "Submodule" not in tb_text


def test_dynamic_submodule_descriptor_on_class_and_method_binding():
    import types

    class DynamicSubmodule(Module):
        def __init__(self):
            super().__init__()

        @dynamic_submodule
        def train(self, *args, **kwargs):
            return {"result": 42}

    assert isinstance(DynamicSubmodule.train, types.FunctionType)

    sub = DynamicSubmodule()
    submodule1 = sub.train
    submodule2 = sub.train

    assert isinstance(submodule1, Module)
    assert submodule1 is submodule2
    assert submodule1({}) == {"result": 42}


def test_dynamic_submodule_nonmethod_exception_stacktrace():
    class DynamicSubmodule(Module):
        def __init__(self):
            super().__init__()

        @dynamic_submodule
        def train(self, design, *args, **kwargs):
            raise ValueError()

    module = DynamicSubmodule()
    wrapped = module.train
    import traceback as _tb
    with pytest.raises(ValueError) as exc_info:
        wrapped({})

    tb_text = "".join(_tb.format_tb(exc_info.value.__traceback__))
    assert "Submodule" not in tb_text


def test_dynamic_submodule_name():
    class DynamicSubmodule(Module):
        def __init__(self):
            super().__init__()

        @dynamic_submodule
        def train(self, *args, **kwargs):
            pass

    module = DynamicSubmodule()
    assert module.train.name == "DynamicSubmodule.train"


def test_dynamic_submodule_name_with_index():
    class DynamicSubmodule(Module):
        def __init__(self):
            super().__init__()

        @dynamic_submodule
        def train(self, *args, **kwargs):
            pass

    module = DynamicSubmodule()
    module.train.i = 0
    assert module.train.name == "DynamicSubmodule.train[0]"


def test_dynamic_submodule_variable_access():
    class DynamicSubmodule(Module):
        def __init__(self):
            super().__init__()
            self.x = 10

        @dynamic_submodule
        def train(self, design, *args, **kwargs):
            return self.x

    module = DynamicSubmodule()
    submodule = module.train
    assert submodule({}) == 10

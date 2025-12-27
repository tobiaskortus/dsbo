from abc import ABC

from dsbo.utils.io import AbstractFilePath


class Module(ABC):
    def __init__(self):
        self._registered_files = {}

    def __call__(self, *args, **kwds):
        pass

    def __setattr__(self, name, value):
        if isinstance(value, AbstractFilePath):
            self._registered_files[name] = value
                
        super().__setattr__(name, value)

    @property
    def registered_files(self) -> dict[str, AbstractFilePath]:
        return self._registered_files


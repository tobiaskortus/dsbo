from enum import Enum
from peewee import CharField


class EnumField(CharField):
    def __init__(self, enum_type: type[Enum], *args, **kwargs):
        self.enum_type = enum_type
        super().__init__(*args, **kwargs)

    def db_value(self, value: Enum) -> str:
        if value is None:
            return None
        if not isinstance(value, self.enum_type):
            raise ValueError(f"Expected value of type {self.enum_type}, got {type(value)}")
        return value.value

    def python_value(self, value: str) -> Enum:
        if value is None:
            return None
        try:
            return self.enum_type(value)
        except ValueError:
            raise ValueError(f"Invalid value '{value}' for enum {self.enum_type}")
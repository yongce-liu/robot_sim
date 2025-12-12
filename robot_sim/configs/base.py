from dataclasses import asdict, dataclass
from enum import Enum
from os import PathLike
from typing import Any

import yaml


class ConfigWrapper:
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, cfg_dict: dict) -> "ConfigWrapper":
        field_types = cls.__annotations__
        kwargs = {}
        for field_name, field_type in field_types.items():
            if field_name not in cfg_dict:
                continue
            value = cfg_dict[field_name]

            # Direct nested config-like class
            if hasattr(field_type, "from_dict"):
                kwargs[field_name] = field_type.from_dict(value)
                continue

            # Direct Enum field
            if isinstance(field_type, type) and issubclass(field_type, Enum):
                kwargs[field_name] = field_type(value)
                continue

            origin = getattr(field_type, "__origin__", None)
            args = getattr(field_type, "__args__", ())

            # List[...] fields
            if origin is list and args:
                inner_type = args[0]
                # List of Enums
                if isinstance(inner_type, type) and issubclass(inner_type, Enum):
                    kwargs[field_name] = [inner_type(item) for item in value]
                # List of nested config-like classes
                elif hasattr(inner_type, "from_dict"):
                    kwargs[field_name] = [inner_type.from_dict(item) for item in value]
                else:
                    kwargs[field_name] = value
                continue
            # Dict[...] fields
            if origin is dict and args:
                key_type, val_type = args
                # Dict with Enum values
                if isinstance(val_type, type) and issubclass(val_type, Enum):
                    kwargs[field_name] = {k: val_type(v) for k, v in value.items()}
                # Dict with nested config-like class values
                elif hasattr(val_type, "from_dict"):
                    kwargs[field_name] = {k: val_type.from_dict(v) for k, v in value.items()}
                else:
                    kwargs[field_name] = value
                continue

            # Optional/Union types where first arg is Enum, e.g. ControlType | None
            if args and any(isinstance(t, type) and issubclass(t, Enum) for t in args if t is not type(None)):
                enum_type = next(t for t in args if t is not type(None) and isinstance(t, type) and issubclass(t, Enum))
                kwargs[field_name] = None if value is None else enum_type(value)
                continue

            # Fallback: keep raw value
            kwargs[field_name] = value
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: PathLike) -> "ConfigWrapper":
        cfg_dict = yaml.safe_load(open(path))
        return cls.from_dict(cfg_dict)

    def save(self, path: PathLike, type="yaml") -> None:
        cfg_dict = self.to_dict()
        with open(path, "w") as f:
            if type == "yaml":
                yaml.dump(cfg_dict, f)
            else:
                raise ValueError(f"Unsupported save type: {type}")

    @staticmethod
    def check_assests():
        pass


def configclass(_cls=None, **kwargs):
    """Decorator to create a dataclass with additional configuration loading and saving methods.
    Args:
        _cls: The class to decorate
    Returns:
        The decorated class with dataclass and config methods.
    """

    def wrap(cls):
        # Create new class inheriting from both cls and ConfigWrapper
        new_cls = type(cls.__name__, (ConfigWrapper,), dict(cls.__dict__))
        new_cls.__module__ = cls.__module__
        new_cls.__qualname__ = cls.__qualname__
        # Apply dataclass decorator
        return dataclass(**kwargs)(new_cls)

    if _cls is None:
        return wrap
    else:
        return wrap(_cls)

from dataclasses import asdict, dataclass
from enum import Enum
from os import PathLike
from typing import Any

import yaml
from loguru import logger


def to_dict(cls) -> dict[str, Any]:
    return asdict(cls)


def from_dict(cls, cfg_dict: dict):
    # Collect annotations from all parent classes (MRO - Method Resolution Order)
    field_types = {}
    for base_cls in reversed(cls.__mro__):
        if hasattr(base_cls, "__annotations__"):
            field_types.update(base_cls.__annotations__)

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


def from_yaml(cls, path: PathLike):
    cfg_dict = yaml.safe_load(open(path))
    cls.print()
    return cls.from_dict(cfg_dict)


def save(cls, path: PathLike, type="yaml") -> None:
    cfg_dict = cls.to_dict()
    with open(path, "w") as f:
        if type == "yaml":
            yaml.dump(cfg_dict, f)
        else:
            raise ValueError(f"Unsupported save type: {type}")


def print(cls) -> None:
    cfg_dict = cls.to_dict()
    yaml_str = yaml.dump(cfg_dict)
    logger.info(f"Configuration:\n{yaml_str}")


_ADD_METHODS = ["to_dict", "from_dict", "from_yaml", "save", "print"]


def configclass(_cls=None, **kwargs):
    """Decorator to create a dataclass with additional configuration loading and saving methods.

    This decorator should be placed at the top (applied last):
        @dataclass
        @other_decorator
        class MyConfig:
            ...

    You can override any method in your class before applying the decorator:
        @dataclass
        class MyConfig:
            def from_dict(cls, cfg_dict: dict):
                # Custom implementation
                ...

    Args:
        _cls: The class to decorate
        **kwargs: Additional arguments to pass to the dataclass decorator
    Returns:
        The decorated class with dataclass and config methods.
    """

    def wrap(cls):
        # Apply dataclass decorator first
        dataclass_cls = dataclass(**kwargs)(cls)

        # Add configuration methods only if they don't already exist
        # This allows users to override them in their class definition
        for method_name in _ADD_METHODS:
            if hasattr(dataclass_cls, method_name):
                continue
            dataclass_cls_method = globals()[method_name]
            setattr(dataclass_cls, method_name, classmethod(dataclass_cls_method))

        return dataclass_cls

    if _cls is None:
        return wrap
    else:
        return wrap(_cls)

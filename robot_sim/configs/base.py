from dataclasses import asdict, dataclass
from enum import Enum
from os import PathLike

import yaml


class ConfigWrapper:
    def to_dict(self) -> dict[str, any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, cfg_dict: dict) -> "ConfigWrapper":
        field_types = cls.__annotations__
        kwargs = {}
        for field_name, field_type in field_types.items():
            if field_name not in cfg_dict:
                continue
            if hasattr(field_type, "from_dict"):
                kwargs[field_name] = field_type.from_dict(cfg_dict[field_name])
            elif getattr(field_type, "__origin__", None) is list:
                if hasattr(field_type.__args__[0], "__mro__") and issubclass(field_type.__args__[0], Enum):
                    kwargs[field_name] = [field_type.__args__[0](item) for item in cfg_dict[field_name]]
                elif hasattr(field_type.__args__[0], "from_dict"):
                    kwargs[field_name] = [field_type.__args__[0].from_dict(item) for item in cfg_dict[field_name]]
                else:
                    kwargs[field_name] = cfg_dict[field_name]
            else:
                kwargs[field_name] = cfg_dict[field_name]
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

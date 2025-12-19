from dataclasses import asdict, dataclass
from enum import Enum
from os import PathLike
from typing import Any

import yaml
from loguru import logger


def _to_dict(cls) -> dict[str, Any]:
    """Convert dataclass instance to dictionary."""
    return asdict(cls)


def _from_dict(cls, cfg_dict: dict):
    """Load dataclass from dictionary using dacite with custom hooks support.

    Uses dacite library for robust dictionary-to-dataclass conversion.

    Supports custom type hooks via class method:
        @classmethod
        def _get_dacite_config(cls) -> dacite.Config:
            return dacite.Config(
                type_hooks={
                    CustomType: custom_hook_function,
                },
                ...
            )

    Automatically handles:
    - Nested config classes with from_dict method
    - Enum types (via cast=[Enum])
    - List/Dict of config classes and enums
    - Optional/Union types
    """
    import dacite

    # Get custom dacite config from subclass if provided
    # Default configuration
    dacite_config = dacite.Config(
        cast=[Enum],
        strict=True,
    )

    if hasattr(cls, "_get_dacite_config"):
        _dacite_config = cls._get_dacite_config()
        for key, value in vars(_dacite_config).items():
            if value is not None:
                setattr(dacite_config, key, value)

    # Use dacite to convert dictionary to dataclass
    # dacite automatically handles nested classes with from_dict methods
    return dacite.from_dict(
        data_class=cls,
        data=cfg_dict,
        config=dacite_config,
    )


def _from_yaml(cls, path: PathLike):
    """Load configuration from YAML file."""
    with open(path) as f:
        cfg_dict = yaml.safe_load(f)
    return cls.from_dict(cfg_dict)


def _save(cls, path: PathLike, save_type: str = "yaml") -> None:
    """Save configuration to file."""
    cfg_dict = cls.to_dict()
    with open(path, "w") as f:
        if save_type == "yaml":
            yaml.dump(cfg_dict, f, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported save type: {save_type}")


def _print(cls) -> None:
    """Print configuration as YAML."""
    cfg_dict = cls.to_dict()
    yaml_str = yaml.dump(cfg_dict, default_flow_style=False)
    logger.info(f"Configuration:\n{yaml_str}")


_CONFIG_METHODS = {
    "to_dict": _to_dict,
    "from_dict": _from_dict,
    "from_yaml": _from_yaml,
    "save": _save,
    "print": _print,
}


def configclass(_cls=None, **dataclass_kwargs):
    """Decorator to create a dataclass with automatic configuration loading.

    This decorator:
    1. Wraps the class with @dataclass
    2. Adds from_dict() method that recursively loads nested config classes
    3. Adds utility methods: to_dict(), from_yaml(), save(), print()

    The from_dict() method automatically handles:
    - Nested config classes (any class with from_dict method)
    - Enum types
    - List[ConfigClass], List[Enum]
    - Dict[K, ConfigClass], Dict[K, Enum]
    - Optional/Union types

    Usage:
        @configclass
        class MyConfig:
            value: int
            nested: NestedConfig
            items: list[ItemConfig]

        config = MyConfig.from_dict({"value": 1, "nested": {...}, "items": [...]})

    You can override any method by defining it in your class:
        @configclass
        class MyConfig:
            @classmethod
            def from_dict(cls, cfg_dict: dict):
                # Custom implementation
                ...

    Args:
        _cls: The class to decorate (automatically passed when used without parentheses)
        **dataclass_kwargs: Additional arguments to pass to @dataclass decorator

    Returns:
        The decorated class with dataclass fields and config methods
    """

    def wrap(cls):
        # Apply dataclass decorator first
        dataclass_cls = dataclass(**dataclass_kwargs)(cls)

        # Add configuration methods only if they don't already exist
        # This allows users to override them in their class definition
        for method_name, method_func in _CONFIG_METHODS.items():
            if not hasattr(dataclass_cls, method_name):
                setattr(dataclass_cls, method_name, classmethod(method_func))
            else:
                logger.debug(f"Method {method_name} already defined in {dataclass_cls.__name__}, skipping addition.")

        return dataclass_cls

    # Support both @configclass and @configclass()
    if _cls is None:
        return wrap
    else:
        return wrap(_cls)

import types
from pathlib import Path
from typing import Union, get_args, get_origin

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf


@pytest.fixture
def project_dir() -> Path:
    """Root directory of the project (repository root)."""

    return Path(__file__).parent.parent


@pytest.fixture
def config_dir(project_dir: Path) -> Path:
    """Path to the configs directory."""
    return project_dir / "configs"


@pytest.fixture
def hydra_config(config_dir: Path) -> DictConfig:
    """Load configuration using Hydra with defaults."""
    with initialize_config_dir(config_dir=str(config_dir.absolute()), version_base=None):
        cfg = compose(config_name="default")
    return cfg


@pytest.fixture
def hydra_config_dict(hydra_config: DictConfig) -> dict:
    """Convert Hydra DictConfig to plain Python dict."""
    return OmegaConf.to_container(hydra_config, resolve=True)


def assert_class_type(obj: object, expect: object) -> None:
    """Assert that an object is of the expected type.

    Args:
        obj: The object to check.
        expect: The expected class/type of the object.
    Raises:
        AssertionError: If the object is not of the expected type.
    """
    for exp_name, exp_type in expect.__annotations__.items():
        obj_value = getattr(obj, exp_name)

        origin = get_origin(exp_type)

        if origin is Union or origin is types.UnionType:
            args = get_args(exp_type)
            if not isinstance(obj_value, args):
                raise AssertionError(f"Field '{exp_name}' is of type {type(obj_value)}, expected {exp_type}.")
        elif origin is list:
            sub_type = get_args(exp_type)[0]
            if hasattr(sub_type, "__annotations__"):
                for item in obj_value:
                    assert_class_type(item, sub_type)
            if not isinstance(obj_value, list):
                raise AssertionError(f"Field '{exp_name}' is of type {type(obj_value)}, expected {exp_type}.")
        elif origin is dict:
            key_type, val_type = get_args(exp_type)
            if hasattr(val_type, "__annotations__"):
                for item in obj_value.values():
                    assert_class_type(item, val_type)
            if not isinstance(obj_value, dict):
                raise AssertionError(f"Field '{exp_name}' is of type {type(obj_value)}, expected {exp_type}.")
        elif origin is not None:
            if not isinstance(obj_value, origin):
                raise AssertionError(f"Field '{exp_name}' is of type {type(obj_value)}, expected {exp_type}.")
        else:
            if not isinstance(obj_value, exp_type):
                raise AssertionError(f"Field '{exp_name}' is of type {type(obj_value)}, expected {exp_type}.")
        if hasattr(exp_type, "__annotations__"):
            assert_class_type(obj_value, exp_type)
        print(f"Field '{exp_name}' passed type check as {exp_type}, with value {obj_value}.")

from dataclasses import MISSING
from typing import Callable

from robot_sim.utils.config import configclass

from .simulator import SimulatorConfig


@configclass
class MapEnvConfig:
    """Configuration for Map Environment."""

    simulator_config: SimulatorConfig = MISSING
    """It can be loaded from a yaml file or defined inline."""
    decimation: int = MISSING
    """Number of simulation steps per environment step."""
    max_episode_steps: int = MISSING
    """Maximum number of steps per episode."""
    observation_map: dict[str, tuple[Callable, dict]] = MISSING
    """
    Mapping of observation group name to a tuple of (callable, config_dict).
    - key (str): Observation group name (e.g., "proprio", "camera", "language")
    - value (tuple): (processing_function, config_parameters)
        - processing_function: Callable that takes "MapEnv" and... returns processed observation dict
        - config_parameters: Dict containing configuration for the processing function
    You can refer to https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/blob/main/unitree_g1.LMPnPAppleToPlateDC/meta/modality.json
    """
    action_map: dict[str, tuple[Callable, dict]] = MISSING
    """
    Mapping of action group name to action processing callable.
    - key (str): Action group name (e.g., "joint_positions", "gripper")
    - value (callable): Function that takes "MapEnv" and returns action dict
    """
    reward_map: dict[str, tuple[Callable, dict]] | None = None
    """Mapping of reward function names to callables and their configurations."""
    termination_map: dict[str, tuple[Callable, dict]] | None = None
    """Mapping of termination condition names to callables and their configurations."""
    truncation_map: dict[str, tuple[Callable, dict]] | None = None
    """Mapping of truncation condition names to callables and their configurations."""

    @staticmethod
    def get_dacite_config():
        import importlib
        from enum import Enum

        import dacite

        def tuple_with_callable_hook(_data: list) -> tuple:
            """Hook to handle tuple[Callable, dict] format in YAML."""
            if isinstance(_data, list) and len(_data) == 2:
                fn, params = _data
                # Check if first item is a Callable spec with _target_
                if isinstance(fn, str):
                    module_path, func_name = fn.rsplit(".", 1)
                    module = importlib.import_module(module_path)
                    func = getattr(module, func_name)
                    return (func, params)
            return tuple(_data)

        dacite_config = dacite.Config(
            type_hooks={
                tuple[Callable, dict]: tuple_with_callable_hook,
            },
            cast=[Enum],
            strict=True,
        )
        return dacite_config


@configclass
class MapTaskConfig:
    """Configuration for Map pick-and-place task."""

    task: str = MISSING
    """Task name for Map environment."""
    env_config: MapEnvConfig = MISSING
    """Map environment configuration."""
    params: dict = MISSING
    """Parameters for the specific task."""

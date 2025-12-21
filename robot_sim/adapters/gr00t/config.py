from dataclasses import MISSING
from typing import Callable

from robot_sim.configs import SimulatorConfig
from robot_sim.utils.config import configclass


@configclass
class Gr00tEnvConfig:
    """Configuration for Gr00t Experiments.

    Attributes:
        robot_name: Name of the robot (e.g., "g1", "gr1")
        enable_gravity_compensation: Whether to enable gravity compensation
        gravity_compensation_joints: List of joint groups for gravity compensation
    """

    simulator_config: SimulatorConfig = MISSING
    """It can be loaded from a yaml file or defined inline."""

    observation_mapping: dict[str, tuple[Callable, dict]] = MISSING
    """
    Mapping of observation group name to a tuple of (callable, config_dict).
    - key (str): Observation group name (e.g., "proprio", "camera", "language")
    - value (tuple): (processing_function, config_parameters)
        - processing_function: Callable that takes "Gr00tEnv" and... returns processed observation dict
        - config_parameters: Dict containing configuration for the processing function
    You can refer to https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim/blob/main/unitree_g1.LMPnPAppleToPlateDC/meta/modality.json
    """
    action_mapping: dict[str, tuple[Callable, dict]] = MISSING
    """
    Mapping of action group name to action processing callable.
    - key (str): Action group name (e.g., "joint_positions", "gripper")
    - value (callable): Function that takes "Gr00tEnv" and returns action dict
    """
    decimation: int = 1
    """Number of simulation steps per environment step."""

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
class Gr00tTaskConfig:
    """Configuration for Gr00t pick-and-place task."""

    task: str = MISSING
    """Task name for Gr00t environment."""
    env_config: Gr00tEnvConfig = MISSING
    """Gr00t environment configuration."""
    params: dict = MISSING
    """Parameters for the specific task."""

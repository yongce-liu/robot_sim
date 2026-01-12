import os
from pathlib import Path
from typing import Any, Callable, Literal, cast

import gymnasium as gym
import numpy as np
import regex as re
import torch
from loguru import logger

from robot_sim.configs import ObjectConfig, RobotModel
from robot_sim.controllers import CompositeController, PIDController


def create_pid_controllers(configs: dict[str, ObjectConfig], dt: float = 0.001) -> dict[str, CompositeController]:
    # Initialize PD controller for low-level control
    controllers = {}
    for name, cfg in configs.items():
        robot = RobotModel(cfg)
        kp = robot.stiffness
        kd = robot.damping
        tor_limits = robot.get_joint_limits("torque", coeff=cfg.extras.get("torque_coeff", 0.9))
        pd_controller = PIDController(kp=kp, kd=kd, dt=dt)
        controllers[name] = CompositeController(
            controllers={"pd": pd_controller}, output_clips={"pd_controller": tor_limits}
        )
    return controllers


def setup_logger(log_file: str, max_file_size: int = 10, mode: str = "w") -> None:
    """
    Setup loguru logger to log to a file in the Hydra output directory.
    Args:
        max_file_size (int): Maximum size of log file in MB before rotation.
    """

    loguru_log_file = f"{log_file}.loguru.log" if not log_file.endswith(".log") else log_file
    max_file_size_b = max_file_size * 1024 * 1024  # Convert MB to bytes
    logger.add(loguru_log_file, rotation=max_file_size_b, mode=mode)


def task_register(
    env_id: str,
    **register_kwargs: Any,
) -> Callable:
    """Decorator to register a task with gymnasium."""

    def decorator(cls):
        entry_point = register_kwargs.pop("entry_point", f"{cls.__module__}:{cls.__name__}")
        if env_id not in gym.registry:
            gym.register(id=env_id, entry_point=entry_point, **register_kwargs)
        cls.env_id = env_id
        return cls

    return decorator


def wrap_array(val: np.ndarray | torch.Tensor | list, device: str = "cpu") -> np.ndarray | torch.Tensor:
    """Get the array type of the simulation."""
    if device == "cpu":
        return np.array(val)
    else:
        return torch.tensor(val, device=device)


def get_env_id(env):
    if hasattr(env, "env_id"):
        return env.env_id
    if hasattr(env, "spec") and env.spec is not None:
        return env.spec.id
    if hasattr(env, "unwrapped") and env.unwrapped.spec:
        return env.unwrapped.spec.id
    return None


def get_reindices(
    source: list[str],
    target: list[str],
    *,
    pattern_position: Literal["source", "target", "none"] = "none",
) -> list[int]:
    result: list[int] = []
    if pattern_position == "source":
        # source: pattern, target: string
        source_rx = [re.compile(p) for p in source]
        for tgt in target:
            result.extend([i for i, rx in enumerate(source_rx) if rx.fullmatch(tgt)])
    elif pattern_position == "target":
        # target: pattern, source: string
        target_rx = [re.compile(p) for p in target]
        for tgt_rx in target_rx:
            result.extend([i for i, s in enumerate(source) if tgt_rx.fullmatch(s)])
    else:
        for tgt in target:
            result.extend([i for i, s in enumerate(source) if s == tgt])

    return result


def resolve_asset_path(path: str | os.PathLike | None) -> str:
    if path is None:
        raise ValueError("Path cannot be None.")
    LOCAL_ASSETS_DIR = Path(__file__).parents[1]
    HF_REPO_ID = "your-username/your-model-repo"
    path_obj = Path(path)

    if path_obj.is_absolute():
        target_path = path_obj
    else:
        target_path = LOCAL_ASSETS_DIR / path_obj

    if target_path.exists():
        return str(target_path)

    logger.info(f"Local path '{target_path}' not found. Attempting download from Hugging Face...")

    try:
        from huggingface_hub import snapshot_download

        folder_pattern = f"{str(path_obj.parent)}/*"
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            allow_patterns=folder_pattern,
            local_dir=LOCAL_ASSETS_DIR / path_obj.parent,
            local_dir_use_symlinks=False,
        )

        if target_path.exists():
            logger.info(f"Assets downloaded to: {target_path.parent}")
            return str(target_path)
        else:
            raise FileNotFoundError(f"Download finished but '{target_path}' is still missing.")

    except Exception as e:
        raise FileNotFoundError(f"Failed to retrieve assets from Hugging Face repo '{HF_REPO_ID}'.") from e


def recursive_setattr(obj: object, props: dict | None, tag: str | None = None) -> None:
    tag = tag if tag is not None else obj.__class__.__name__

    if props is None or len(props) == 0:
        logger.warning(f"You provided empty properties for {tag}")
        return

    def _recursive(o: object, p: dict):
        for key, value in p.items():
            if isinstance(value, dict) and hasattr(o, key):
                _recursive(getattr(o, key), value)
            else:
                try:
                    setattr(o, key, value)
                except AttributeError:
                    logger.error(f"Unknown option '{key}' ignored.")

    _recursive(obj, props)

    logger.info(f"Successfully set properties for {tag}: {props}")


class CommandSmoother:
    """
    Smooth and validate joint commands by limiting velocity and acceleration.

    This class ensures safe robot operation by preventing sudden movements that could
    damage hardware or cause instability. It applies velocity and acceleration limits
    to incoming joint position commands.
    """

    def __init__(
        self,
        dt: float,
        enable: bool = True,
        alpha: np.ndarray | float = 1.0,
        max_velocity: np.ndarray | float = float("inf"),
        initial_positions: np.ndarray | None = None,
    ):
        """
        Initialize the command smoother.

        Args:
            alpha: Smoothing factor between 0 and 1 (1 = no smoothing)
            initial_positions: Initial joint positions array of shape (..., num_joints)
            dt: Control time step in seconds
            max_velocity: Maximum allowed joint velocity in rad/s
            enable: Whether to enable smoothing (if False, commands pass through unchanged)
        """
        self.dt = dt
        self.max_velocity = max_velocity
        self.max_delta_pos = max_velocity * dt
        self.alpha = alpha
        self.enable = enable

        # State tracking
        self.last_targets: np.ndarray | None = None
        if initial_positions is not None:
            self.reset(initial_positions)

    def reset(self, initial_positions: np.ndarray) -> None:
        """
        Reset the smoother state with initial joint positions.

        Args:
            initial_positions: Initial joint positions array of shape (..., num_joints)
        """
        self.last_targets = initial_positions.copy()

    def smooth(self, target_positions: np.ndarray) -> np.ndarray:
        """
        Apply smoothing to target joint positions.

        Args:
            target_positions: Desired joint positions of shape (..., num_joints)

        Returns:
            Smoothed joint positions with velocity and acceleration limits applied
        """
        if not self.enable:
            return target_positions

        if self.last_targets is None:
            self.reset(target_positions)
            return target_positions

        target_positions = (1 - self.alpha) * self.last_targets + self.alpha * target_positions

        # Calculate position change && expected velocity && clip to max delta position
        delta_pos = (target_positions - self.last_targets).clip(-self.max_delta_pos, self.max_delta_pos)
        self.last_targets += delta_pos

        return cast(np.ndarray, self.last_targets)

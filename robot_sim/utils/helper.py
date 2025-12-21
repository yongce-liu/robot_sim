from typing import Any, Callable

import gymnasium as gym


def setup_logger(max_file_size: int = 10, mode: str = "w") -> None:
    """
    Setup loguru logger to log to a file in the Hydra output directory.
    Args:
        max_file_size (int): Maximum size of log file in MB before rotation.
    """
    from hydra.core.hydra_config import HydraConfig
    from loguru import logger

    hydra_cfg = HydraConfig.get()
    loguru_log_file = f"{hydra_cfg.runtime.output_dir}/{hydra_cfg.job.name}.loguru.log"
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
        cls.gym_id = env_id
        return cls

    return decorator

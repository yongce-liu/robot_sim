from typing import Any, Callable, Literal

import gymnasium as gym
import regex as re


def setup_logger(log_file: str, max_file_size: int = 10, mode: str = "w") -> None:
    """
    Setup loguru logger to log to a file in the Hydra output directory.
    Args:
        max_file_size (int): Maximum size of log file in MB before rotation.
    """
    from loguru import logger

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
        cls.gym_id = env_id
        return cls

    return decorator


def get_reindices(
    source: list[str],
    target: list[str],
    *,
    pattern_position: Literal["source", "target", "none"] = "none",
) -> list[list[int]]:
    if pattern_position == "source":
        source = [re.compile(p) for p in source]

    if pattern_position == "target":
        target = [re.compile(p) for p in target]

    result: list[list[int]] = []

    for tgt in target:
        if pattern_position == "source":
            # source: pattern, target: string
            matched = [i for i, rx in enumerate(source) if rx.fullmatch(tgt)]
        elif pattern_position == "target":
            # target: pattern, source: string
            matched = [i for i, s in enumerate(source) if tgt.fullmatch(s)]
        else:
            matched = [i for i, s in enumerate(source) if s == tgt]

        result.extend(matched)

    return result

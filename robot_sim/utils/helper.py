import os
from pathlib import Path
from typing import Any, Callable, Literal

import gymnasium as gym
import regex as re
from loguru import logger


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


def resolve_asset_path(path: str | os.PathLike) -> str:
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

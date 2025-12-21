from typing import Any, Callable

from gymnasium.envs import registration

_REGISTERED_TASKS: list[str] = []


def task_register(env_id: str, **register_kwargs: Any) -> Callable:
    """Decorator to register a task with gymnasium."""

    def decorator(cls):
        entry_point = register_kwargs.pop("entry_point", f"{cls.__module__}:{cls.__name__}")
        if env_id not in registration.registry:
            registration.register(id=env_id, entry_point=entry_point, **register_kwargs)
        _REGISTERED_TASKS.append(env_id)
        cls.gym_id = env_id
        return cls

    return decorator


def registered_tasks() -> list[str]:
    from robot_sim.adapters.gr00t.tasks import pick_place  # noqa: F401

    return _REGISTERED_TASKS

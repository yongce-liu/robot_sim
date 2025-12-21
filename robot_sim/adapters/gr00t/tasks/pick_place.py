"""Pick-and-place task for Gr00t adapter."""

from typing import Any

import numpy as np

from robot_sim.adapters.gr00t import Gr00tEnv, Gr00tEnvConfig
from robot_sim.adapters.gr00t.tasks import task_register


@task_register("Gr00tPickAndPlace-v0")
class PickAndPlaceTask(Gr00tEnv):
    """Pick-and-place task wrapper.

    This wrapper computes task-specific reward, termination, truncation, and info
    based on object and target positions.
    """

    def __init__(
        self,
        env_config: Gr00tEnvConfig,
        object_name: str | None = None,
        target_name: str | None = None,
        target_pos: np.ndarray | list[float] | None = None,
        object_obs_key: str | None = None,
        target_obs_key: str | None = None,
        success_threshold: float = 0.05,
        dense_reward: bool = True,
        reward_scale: float = 1.0,
        max_episode_steps: int = 200,
    ) -> None:
        super().__init__(env_config)

        if target_name is None and target_pos is None and target_obs_key is None:
            raise ValueError("Provide target_name, target_pos, or target_obs_key for PickAndPlaceEnv")
        if object_name is None and object_obs_key is None:
            raise ValueError("Provide object_name or object_obs_key for PickAndPlaceEnv")

        self.object_name = object_name
        self.target_name = target_name
        self.target_pos = None if target_pos is None else np.asarray(target_pos, dtype=np.float32)
        self.object_obs_key = object_obs_key
        self.target_obs_key = target_obs_key
        self.success_threshold = float(success_threshold)
        self.dense_reward = dense_reward
        self.reward_scale = float(reward_scale)
        self.max_episode_steps = int(max_episode_steps)

        self._episode_step = 0
        self._last_distance: np.ndarray | float | None = None

    def reset(self, *args: Any, **kwargs: Any):
        self._episode_step = 0
        return self.env.reset(*args, **kwargs)

    def step(self, action: Any):
        observation, _, _, _, info = self.env.step(action)
        self._episode_step += 1

        reward = self.compute_reward(observation, action)
        terminated = self.compute_terminated(observation, action)
        truncated = self.compute_truncated(observation, action)

        task_info = self.compute_info(observation, action)
        if info is None:
            info = task_info
        else:
            info = {**info, **task_info}

        return observation, reward, terminated, truncated, info

    def compute_reward(self, observation: Any, action: Any | None = None) -> float | np.ndarray:
        distance = self._compute_distance(observation)
        self._last_distance = distance
        if self.dense_reward:
            reward = -distance
        else:
            reward = (distance <= self.success_threshold).astype(np.float32)
        return reward * self.reward_scale

    def compute_terminated(self, observation: Any, action: Any | None = None) -> bool | np.ndarray:
        distance = self._compute_distance(observation)
        return distance <= self.success_threshold

    def compute_truncated(self, observation: Any, action: Any | None = None) -> bool | np.ndarray:
        return self._episode_step >= self.max_episode_steps

    def compute_info(self, observation: Any, action: Any | None = None) -> dict[str, Any]:
        distance = self._last_distance
        if distance is None:
            distance = self._compute_distance(observation)
        return {
            "distance": distance,
            "success": distance <= self.success_threshold,
        }

    def _compute_distance(self, observation: Any) -> np.ndarray | float:
        object_pos = self._get_object_pos(observation)
        target_pos = self._get_target_pos(observation)
        return np.linalg.norm(object_pos - target_pos, axis=-1)

    def _get_object_pos(self, observation: Any) -> np.ndarray:
        if self.object_obs_key and isinstance(observation, dict) and self.object_obs_key in observation:
            return self._extract_pos(observation[self.object_obs_key])
        if self.object_name is None:
            raise ValueError("object_name is required when object_obs_key is not provided")
        return self._get_state_pos(self.object_name)

    def _get_target_pos(self, observation: Any) -> np.ndarray:
        if self.target_obs_key and isinstance(observation, dict) and self.target_obs_key in observation:
            return self._extract_pos(observation[self.target_obs_key])
        if self.target_name is not None:
            return self._get_state_pos(self.target_name)
        if self.target_pos is None:
            raise ValueError("target_pos is required when target_name and target_obs_key are not provided")
        return np.asarray(self.target_pos, dtype=np.float32)

    def _get_state_pos(self, name: str) -> np.ndarray:
        if not hasattr(self.env, "backend"):
            raise AttributeError("Wrapped env has no backend to query object state")
        states = self.env.backend.get_states()
        if name not in states.objects:
            raise KeyError(f"Object '{name}' not found in backend states")
        root_state = self._to_numpy(states.objects[name].root_state)
        return root_state[..., :3]

    @staticmethod
    def _extract_pos(value: Any) -> np.ndarray:
        arr = PickAndPlaceEnv._to_numpy(value)
        if arr.shape[-1] < 3:
            raise ValueError(f"Position source has invalid shape {arr.shape}; expected last dim >= 3")
        return arr[..., :3]

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

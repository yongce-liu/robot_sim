"""Pick-and-place task for Gr00t adapter."""

from typing import Any

import numpy as np
from loguru import logger

from robot_sim.adapters.gr00t.env import Gr00tWBCEnv
from robot_sim.backends.types import ObjectState
from robot_sim.configs import MapTaskConfig
from robot_sim.utils.helper import task_register


@task_register("Gr00tPickAndPlace-v0")
class PickAndPlaceTask(Gr00tWBCEnv):
    """Pick-and-place task wrapper.

    This wrapper computes task-specific reward, termination, truncation, and info
    based on object and target positions.
    """

    def __init__(
        self,
        env_config: MapTaskConfig,
        object_name: str,
        target_name: str,
        success_threshold: float = 1e-2,
    ) -> None:
        super().__init__(env_config)
        logger.info(f"PickAndPlaceTask with object: {object_name}, target: {target_name}, robot: {self.robot_name}")

        self.object_name = object_name
        self.target_name = target_name
        self.success_threshold = float(success_threshold)

        self._distance_cache: np.ndarray | None = None
        self._distance_cache_expire: bool = True

    def reset(self, **kwargs) -> Any:
        self._distance_cache_expire = True
        return super().reset(**kwargs)

    def step(self, action: Any) -> Any:
        self._distance_cache_expire = True
        return super().step(action)

    def compute_terminated(self, observation, action=None):
        is_success = self.obj_tgt_distance < self.success_threshold
        return is_success

    def compute_reward(self, observation, action=None):
        reward = -self.obj_tgt_distance * self.reward_scale

        return reward

    def compute_info(self, observation, action=None):
        info = super().compute_info(observation, action)
        is_success = info.get("is_success", None)
        if is_success is None:
            is_success = self.obj_tgt_distance < self.success_threshold
            info["is_success"] = is_success
        return info

    def get_object_state(self, name: str | None = None) -> ObjectState:
        """Get the object position used in the task.

        Returns:
            Object position as a numpy array.
        """
        if name is None:
            name = self.object_name
        states = self.get_states(name)
        return states

    def get_target_state(self, name: str | None = None) -> ObjectState:
        """Get the target position used in the task.

        Returns:
            Target position as a numpy array, or None if not specified.
        """
        if name is None:
            name = self.target_name
        states = self.get_states(name)
        return states

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @property
    def obj_tgt_distance(self) -> np.ndarray:
        """Get the distance between the object and target.

        Returns:
            Distance as a numpy array.
        """
        if self._distance_cache_expire:
            object_pos = self.get_object_state().root_state[:3]
            target_pos = self.get_target_state().root_state[:3]
            self._distance_cache = np.linalg.norm(object_pos - target_pos, axis=-1)
            self._distance_cache_expire = False
        return self._distance_cache

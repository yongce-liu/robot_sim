"""Pick-and-place task for Gr00t adapter."""

from typing import Any

import numpy as np
from loguru import logger

from robot_sim.adapters.gr00t import Gr00tWBCEnv
from robot_sim.backends.types import ArrayState, ObjectState
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

        max_episode_steps: int = (200,)

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

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def get_object_state(self, states: ArrayState) -> ObjectState:
        """Get the object position used in the task.

        Returns:
            Object position as a numpy array.
        """
        return states.objects[self.object_name]

    def get_target_state(self, states: ArrayState) -> ObjectState:
        """Get the target position used in the task.

        Returns:
            Target position as a numpy array, or None if not specified.
        """
        return states.objects[self.target_name]

"""Stack task for Gr00t adapter."""

from typing import Any

import numpy as np
from loguru import logger

from robot_sim.adapters.gr00t.env import Gr00tEnv
from robot_sim.configs import SimulatorConfig
from robot_sim.utils.helper import task_register


@task_register("Gr00tStack-v0")
class StackTask(Gr00tEnv):
    """
    Stack task wrapper.
    This wrapper computes task-specific reward, termination, truncation, and info
    based on object and target positions.
    """

    def __init__(
        self,
        config: SimulatorConfig,
        object_name: str,
        target_name: str,
        success_threshold: float = 1e-2,
        reward_scale: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(config=config, **kwargs)
        logger.info(f"PickAndPlaceTask with object: {object_name}, target: {target_name}, robot: {self.robot_name}")

        self.object_name = object_name
        self.target_name = target_name
        obj_size = self.get_object_config(object_name).properties.get("size", [0.05] * 3)
        tgt_size = self.get_object_config(target_name).properties.get("size", [0.05] * 3)
        self.success_threshold = np.array(
            [success_threshold] * 2 + [obj_size[2] / 2 + tgt_size[2] / 2 + success_threshold]
        )
        self.reward_scale = float(reward_scale)

        self._diff_cache: np.ndarray | None = None
        self._diff_cache_expire: bool = True

    def reset(self, **kwargs):
        self._diff_cache_expire = True
        return super().reset(**kwargs)

    def step(self, action: Any) -> Any:
        self._diff_cache_expire = True
        return super().step(action)

    def compute_terminated(self, observation, action=None):
        is_success = np.all(self.obj_tgt_diff <= self.success_threshold)
        return is_success

    def compute_reward(self, observation, action=None):
        reward = -np.sum(self.obj_tgt_diff) * self.reward_scale

        return reward.item()

    def compute_info(self, observation, action=None):
        info = super().compute_info(observation, action)
        is_success = info.get("is_success", None)
        if is_success is None:
            is_success = np.all(self.obj_tgt_diff <= self.success_threshold)
            info["success"] = is_success
        return info

    @property
    def obj_tgt_diff(self) -> np.ndarray:
        """Get the distance between the object and target.

        Returns:
            Difference as a numpy array.
        """
        if self._diff_cache_expire:
            object_pos = self.get_object_state(self.object_name).root_state[..., :3]
            target_pos = self.get_object_state(self.target_name).root_state[..., :3]
            self._diff_cache = np.abs(object_pos - target_pos)
            self._diff_cache_expire = False
        if self._diff_cache is None:
            raise ValueError("Difference cache is None. This should not happen.")
        return self._diff_cache

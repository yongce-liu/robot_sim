"""Pick-and-place task for Gr00t adapter."""

from typing import Any

import numpy as np
from loguru import logger

from robot_sim.adapters.gr00t.env import Gr00tEnv
from robot_sim.configs import SimulatorConfig
from robot_sim.utils.helper import task_register


@task_register("Gr00tPickAndPlace-v0")
class PickAndPlaceTask(Gr00tEnv):
    """Pick-and-place task wrapper.

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
        self.success_threshold = float(success_threshold)
        self.reward_scale = float(reward_scale)

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
        object_pos = self.get_object_state(self.object_name).root_state[...,:3]
        return is_success.item() or np.all(object_pos[..., 2] < 0.5) or np.all(object_pos[..., 0] > 0.4)

    def compute_reward(self, observation, action=None):
        reward = -self.obj_tgt_distance * self.reward_scale

        return reward.item()

    def compute_info(self, observation, action=None):
        info = super().compute_info(observation, action)
        is_success = info.get("is_success", None)
        if is_success is None:
            is_success = self.obj_tgt_distance < self.success_threshold
            info["success"] = is_success
        return info

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
            object_pos = self.get_object_state(self.object_name).root_state[..., :3]
            target_pos = self.get_object_state(self.target_name).root_state[..., :3]
            self._distance_cache = np.linalg.norm(object_pos - target_pos, axis=-1)
            self._distance_cache_expire = False
        if self._distance_cache is None:
            raise ValueError("Distance cache is None. This should not happen.")
        return self._distance_cache

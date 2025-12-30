from dataclasses import MISSING, dataclass
from typing import Any, Callable

import gymnasium as gym
import numpy as np

from robot_sim.backends.types import ActionsType, StatesType
from robot_sim.configs import SimulatorConfig
from robot_sim.controllers import CompositeController
from robot_sim.envs.base import BaseEnv


@dataclass
class MapCache:
    observation: dict[str, Callable] = MISSING
    """
    Mapping of observation group name to a tuple of (callable, config_dict).
    - key (str): Observation group name (e.g., "proprio", "camera", "language")
    - value (tuple): config_parameters: Dict containing configuration for the processing function
    """
    action: dict[str, Callable] = MISSING
    """
    Mapping of action group name to action processing callable.
    - key (str): Action group name (e.g., "joint_positions", "gripper")
    - value (callable): Function that takes ""MapEnv"" and returns action dict
    """
    reward: dict[str, Callable] | None = None
    """Mapping of reward function names to callables and their configurations."""
    termination: dict[str, Callable] | None = None
    """Mapping of termination condition names to callables and their configurations."""
    truncation: dict[str, Callable] | None = None
    """Mapping of truncation condition names to callables and their configurations."""
    info: dict[str, Callable] | None = None
    """Mapping of info function names to callables and their configurations."""


class MapEnv(BaseEnv, gym.Env):
    """Environment wrapper for BaseEnv.

    This environment wraps the backend simulator and provides an interface
    compatible with whole body control framework.

    Args:
        config: Simulator configuration.
        **kwargs: Additional configuration options, e.g, render_mode
    Note:
        Only single robot and single environment are supported in MapEnv.
        You need to ihherit this class and implement the following abstract methods:
        - _init_controller: Initialize the composite controller for the robot.
        - _init_maps: Initialize the observation, action, reward, termination, truncation, and info maps.
    """

    def __init__(
        self,
        config: SimulatorConfig,
        **kwargs,
    ) -> None:
        super().__init__(config=config, **kwargs)
        assert config.sim.num_envs == 1, "Only single environment supported in MapEnv currently."
        assert len(self.robot_names) == 1, "Only single robot supported in MapEnv currently."
        self._observation_space_dict: dict[str, gym.spaces.Space] = {}
        self._action_space_dict: dict[str, gym.spaces.Space] = {}
        self._controller: CompositeController | None = None
        self._map_cache: MapCache | None = None

    def statesType2observation(self, states: StatesType) -> gym.spaces.Dict:
        observation_dict = {}
        for name in self.robot_names:
            for group_name, map_func in self.observation_map.items():
                res = map_func(name=name, states=states)
                # assert res is not None, f"Observation map function for group '{group_name}' returned None."
                if res is not None:
                    observation_dict[group_name] = res

        return observation_dict

    def action2actionsType(self, action: dict[str, Any]) -> ActionsType:
        """Convert action to backend format.

        Args:
            action: Action dictionary with key group names and values

        Returns:
            Action dictionary in backend format with key as robot name and value as action array (torque control currently, position control may be added later)
        """
        output: ActionsType = {}
        for name in self.robot_names:
            action[name] = np.full(
                shape=(self.num_envs, self.num_dofs[name]), fill_value=np.nan, dtype=np.float32
            )  # It is used to output the final action array (ActionsType) for the backend
            for group_name, map_func in self.action_map.items():
                res = map_func(name=name, action=action, states=self.states)
                if res is not None:
                    action[group_name] = res
            res: ActionsType = self.controller.compute(name=name, states=self.states, targets=action)
            assert len(res) == 1, f"Action map function for robot '{name}' returned multiple action arrays."
            output[name] = res[name]
        return output

    def compute_reward(self, observation, action=None):
        reward = 0.0
        for group_name, map_func in self.reward_map.items():
            res = map_func(observation=observation, action=action)
            if res is not None:
                reward += res
        return reward

    def compute_terminated(self, observation, action=None):
        terminated = False
        for group_name, map_func in self.termination_map.items():
            res = map_func(observation=observation, action=action)
            if res is not None:
                terminated |= res
        return terminated

    def compute_truncated(self, observation, action=None):
        truncated = self.episode_step >= self.max_episode_steps
        for group_name, map_func in self.truncation_map.items():
            res = map_func(observation=observation, action=action)
            if res is not None:
                truncated |= res
        return truncated

    def compute_info(self, observation, action=None):
        info = {}
        for group_name, map_func in self.info_map.items():
            res = map_func(observation=observation, action=action)
            if res is not None:
                info.update(res)
        return info

    @property
    def observation_space(self) -> gym.spaces.Dict:
        assert len(self._observation_space_dict) > 0, "Observation space not initialized."
        return gym.spaces.Dict(self._observation_space_dict)

    @property
    def action_space(self) -> gym.spaces.Dict:
        assert len(self._action_space_dict) > 0, "Action space not initialized."
        return gym.spaces.Dict(self._action_space_dict)

    @property
    def controller(self) -> CompositeController:
        assert self._controller is not None, "Controller not initialized."
        return self._controller

    @property
    def map_cache(self) -> MapCache:
        assert self._map_cache is not None, "Map cache not initialized."
        return self._map_cache

    @property
    def robot_name(self) -> str:
        return self.robot_names[0]

    @property
    def observation_map(self) -> dict[str, Callable]:
        return self._map_cache.observation if self._map_cache.observation is not None else {}

    @property
    def action_map(self) -> dict[str, Callable]:
        return self._map_cache.action if self._map_cache.action is not None else {}

    @property
    def reward_map(self) -> dict[str, Callable]:
        return self._map_cache.reward if self._map_cache.reward is not None else {}

    @property
    def termination_map(self) -> dict[str, Callable]:
        return self._map_cache.termination if self._map_cache.termination is not None else {}

    @property
    def truncation_map(self) -> dict[str, Callable]:
        return self._map_cache.truncation if self._map_cache.truncation is not None else {}

    @property
    def info_map(self) -> dict[str, Any]:
        return self._map_cache.info if self._map_cache.info is not None else {}

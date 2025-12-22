from abc import abstractmethod
from typing import Any

import gymnasium as gym
import numpy as np
from loguru import logger

from robot_sim.backends.types import ActionsType, StatesType
from robot_sim.configs import MapEnvConfig, MapFunc, ObjectType
from robot_sim.controllers import CompositeController
from robot_sim.envs.base import BaseEnv


class MapEnv(BaseEnv, gym.Env):
    """Environment wrapper for BaseEnv.

    This environment wraps the backend simulator and provides an interface
    compatible with whole body control framework.

    Args:
        config: Speicialized configuration for MapEnv
        **kwargs: Additional configuration options, e.g, render_mode
    """

    def __init__(
        self,
        config: MapEnvConfig | None = None,
        **kwargs,
    ) -> None:
        _robot_names = [
            obj_name
            for obj_name, obj_cfg in config.simulator_config.scene.objects.items()
            if obj_cfg.type == ObjectType.ROBOT
        ]

        assert len(_robot_names) == 1, "Only single robot supported in MapEnv"
        assert config.simulator_config.sim.num_envs == 1, "Only single environment supported in MapEnv"
        super().__init__(config=config.simulator_config, decimation=config.decimation, **kwargs)
        self.config = config
        self._robot_name = _robot_names[0]
        self.robot_cfg = config.simulator_config.scene.objects[self.robot_name]
        self._max_episode_steps = config.max_episode_steps
        self._episode_step = 0

        #################################################################
        self._observation_space_dict: dict[str, gym.spaces.Space] = {}
        self._action_space_dict: dict[str, gym.spaces.Space] = {}
        self._num_dofs = len(self.backend.get_actuator_names(self.robot_name))

        ############# Initialize observation and action map #############
        logger.info("Initializing maps of observation, action, reward, termination, truncation, and info...")
        for _map_dict in [
            self.observation_map,
            self.action_map,
            self.reward_map,
            self.termination_map,
            self.truncation_map,
            self.info_map,
        ]:
            for _group_name, (_map_func, _params) in _map_dict.items():
                _map_func.init(self, _group_name, **_params)

        logger.info(
            f"Maps initialized.\nObservation Space: {self.observation_space}\nAction Space: {self.action_space}"
        )

        self.controller: CompositeController = self._init_controller()

    def reset(self, *, seed=None, options=None):
        self._episode_step = 0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        self._episode_step += 1
        return super().step(action)

    def statesArray2observation(self, states: StatesType) -> gym.spaces.Dict:
        observation_dict = {}
        for group_name, (map_func, params) in self.observation_map.items():
            res = map_func(states=states, **params)
            assert res is not None, f"Observation map function for group '{group_name}' returned None."
            observation_dict[group_name] = res

        return observation_dict

    def action2actionArray(self, action: dict[str, Any]) -> ActionsType:
        """Convert action to backend format.

        Args:
            action: Action dictionary with key group names and values

        Returns:
            Action dictionary in backend format with key as robot name and value as action array (torque control currently, position control may be added later)
        """
        action[self.robot_name] = np.zeros(
            shape=(1, self.num_dofs), dtype=np.float32
        )  # It is used to output the final action array (ActionsType) for the backend
        for group_name, (map_func, params) in self.action_map.items():
            res = map_func(action=action, **params)
            if res is not None:
                action[group_name] = res
        states = self.get_states(self.robot_name)
        action_array = self.controller.compute(states=states, action=action)
        return action_array

    def compute_reward(self, observation, action=None):
        reward = 0.0
        for group_name, (map_func, params) in self.reward_map.items():
            res = map_func(observation=observation, action=action, **params)
            if res is not None:
                reward += res
        return reward

    def compute_terminated(self, observation, action=None):
        terminated = False
        for group_name, (map_func, params) in self.termination_map.items():
            res = map_func(observation=observation, action=action, **params)
            if res is not None:
                terminated |= res
        return terminated

    def compute_truncated(self, observation, action=None):
        truncated = self.episode_step >= self.max_episode_steps
        for group_name, (map_func, params) in self.truncation_map.items():
            res = map_func(observation=observation, action=action, **params)
            if res is not None:
                truncated |= res
        return truncated

    def compute_info(self, observation, action=None):
        info = {}
        for group_name, (map_func, params) in self.info_map.items():
            res = map_func(observation=observation, action=action, **params)
            if res is not None:
                info.update(res)
        return info

    @abstractmethod
    def _init_controller(self) -> CompositeController:
        raise NotImplementedError("CompositeController initialization not implemented yet.")

    @property
    def observation_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict(self._observation_space_dict)

    @property
    def action_space(self) -> gym.spaces.Dict:
        return gym.spaces.Dict(self._action_space_dict)

    @property
    def episode_step(self) -> int:
        return self._episode_step

    @episode_step.setter
    def episode_step(self, value: int) -> None:
        self._episode_step = value

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps

    @property
    def robot_name(self) -> str:
        return self._robot_name

    @property
    def num_dofs(self) -> int:
        return self._num_dofs

    @property
    def observation_map(self) -> dict[str, tuple[MapFunc, dict]]:
        return self.config.observation_map if self.config.observation_map is not None else {}

    @property
    def action_map(self) -> dict[str, tuple[MapFunc, dict]]:
        return self.config.action_map if self.config.action_map is not None else {}

    @property
    def reward_map(self) -> dict[str, tuple[MapFunc, dict]]:
        return self.config.reward_map if self.config.reward_map is not None else {}

    @property
    def termination_map(self) -> dict[str, tuple[MapFunc, dict]]:
        return self.config.termination_map if self.config.termination_map is not None else {}

    @property
    def truncation_map(self) -> dict[str, tuple[MapFunc, dict]]:
        return self.config.truncation_map if self.config.truncation_map is not None else {}

    @property
    def info_map(self) -> dict[str, tuple[MapFunc, dict]]:
        return self.config.info_map if self.config.info_map is not None else {}

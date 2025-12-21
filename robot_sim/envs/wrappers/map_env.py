from abc import abstractmethod
from typing import Any, Callable

import gymnasium as gym
from loguru import logger

from robot_sim.backends.types import ActionType, ArrayState
from robot_sim.configs import MapEnvConfig, ObjectType
from robot_sim.controllers import CompositeController
from robot_sim.envs.base import BaseEnv


class MapEnv(BaseEnv):
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

        ############# Initialize observation and action map #############
        self._observation_map: dict[str, Callable[[str, "MapEnv"]]] = {}
        self._action_map: dict[str, Callable[[str, "MapEnv"]]] = {}
        self._reward_map: dict[str, Callable[[str, "MapEnv"]]] = {}
        self._termination_map: dict[str, Callable[[str, "MapEnv"]]] = {}
        self._truncation_map: dict[str, Callable[[str, "MapEnv"]]] = {}
        self._info_map: dict[str, Callable[[str, "MapEnv"]]] = {}
        #################################################################
        self._observation_space_dict: dict[str, gym.spaces.Space] = {}
        self._action_space_dict: dict[str, gym.spaces.Space] = {}
        self._num_dofs = len(self.backend.get_actuator_names(self.robot_name))

        logger.info("Initializing observation and action map...")
        self._init_observation_map_space()
        self._init_action_map_space()
        logger.info(f"Observation Space: {self.observation_space}\nAction Space: {self.action_space}")

        self.controller: CompositeController = self._init_controller()

    def reset(self, *, seed=None, options=None):
        self._episode_step = 0
        return super().reset(seed=seed, options=options)

    def step(self, action):
        self._episode_step += 1
        return super().step(action)

    def stateArray2observation(self, states: ArrayState) -> gym.spaces.Dict:
        observation_dict = {}
        for group_name, (callable_fn, params) in self._observation_map.items():
            res = callable_fn(group_name, self, **params, states=states)
            if res is not None:
                observation_dict[group_name] = res
        return observation_dict

    def action2actionArray(self, action: dict[str, Any]) -> ActionType:
        """Convert action to backend format.

        Args:
            action: Action dictionary with key group names and values

        Returns:
            Action dictionary in backend format with key as robot name and value as action array (torque control currently, position control may be added later)
        """
        for group_name, (callable_fn, params) in self._action_map.items():
            res = callable_fn(group_name, self, **params, action=action)
            if res is not None:
                action[group_name] = res
        action_array = self.controller.compute(action)
        return action_array

    def compute_reward(self, observation, action=None):
        reward = 0.0
        for group_name, (callable_fn, params) in self.config.reward_map.items():
            res = callable_fn(group_name, self, **params, observation=observation, action=action)
            if res is not None:
                reward += res
        return reward

    def compute_terminated(self, observation, action=None):
        terminated = False
        for group_name, (callable_fn, params) in self.config.termination_map.items():
            res = callable_fn(group_name, self, **params, observation=observation, action=action)
            if res is not None:
                terminated |= res
        return terminated

    def compute_truncated(self, observation, action=None):
        truncated = self.episode_step >= self.max_episode_steps
        for group_name, (callable_fn, params) in self.config.truncation_map.items():
            res = callable_fn(group_name, self, **params, observation=observation, action=action)
            if res is not None:
                truncated |= res
        return truncated

    def compute_info(self, observation, action=None):
        info = {}
        for group_name, (callable_fn, params) in self.config.info_map.items():
            res = callable_fn(group_name, self, **params, observation=observation, action=action)
            if res is not None:
                info.update(res)
        return info

    def _init_observation_map_space(self) -> None:
        for group_name, (callable_fn, params) in self.config.observation_map.items():
            callable_fn(group_name, self, **params)
            self._observation_map[group_name] = (callable_fn, params)

    def _init_action_map_space(self) -> None:
        for group_name, (callable_fn, params) in self.config.action_map.items():
            callable_fn(group_name, self, **params)
            self._action_map[group_name] = (callable_fn, params)

    @abstractmethod
    def _init_controller(self) -> CompositeController:
        raise NotImplementedError("CompositeController initialization not implemented yet.")

    @property
    def observation_map(self) -> dict[str, Callable]:
        return self._observation_map

    @property
    def action_map(self) -> dict[str, Callable]:
        return self._action_map

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

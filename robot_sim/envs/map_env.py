from dataclasses import dataclass
from typing import Any, Callable

import gymnasium as gym
import numpy as np

from robot_sim.backends.types import ActionsType, StatesType
from robot_sim.configs import SimulatorConfig
from robot_sim.controllers import CompositeController, PIDController
from robot_sim.envs.base import BaseEnv


@dataclass
class MapCache:
    observation: dict[str, Callable]
    """
    Mapping of observation group name to a tuple of (callable, config_dict).
    - key (str): Observation group name (e.g., "proprio", "camera", "language")
    - value (tuple): config_parameters: Dict containing configuration for the processing function
    """
    action: dict[str, Callable]
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
        assert self.num_envs == 1, "Only single environment supported in MapEnv currently."
        # Initialize Gym spaces early so map builders can populate them.
        self.observation_space: gym.spaces.Dict = gym.spaces.Dict({})
        self.action_space: gym.spaces.Dict = gym.spaces.Dict({})
        self._map_cache: MapCache | None = None

    def reset(  # gym-like signature
        self,
        states: StatesType | None = None,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset the environment to an initial state.

        Args:
            states: Optional initial states to reset the environment to.
            seed: Optional seed for random number generators.
            options: Optional dictionary of additional options for resetting the environment.

        Returns:
            observation: The initial observation.
            info: Additional information dictionary.
        """
        gym.Env.reset(self, seed=seed, options=options)
        # Reset the backend state
        if states is None:
            states = options.get("initial_states", self.initial_states) if options is not None else self.initial_states

        return BaseEnv.reset(self, states=states)

    def create_controllers(self, coeff: float = 0.9, **kwargs) -> dict[str, CompositeController]:
        # Initialize PD controller for low-level control
        controllers = {}
        for name, robot in self.robots.items():
            kp = robot.stiffness
            kd = robot.damping
            tor_limits = robot.get_joint_limits("torque", coeff=coeff)
            pd_controller = PIDController(kp=kp, kd=kd, dt=self.step_dt / self.decimation)
            controllers[name] = CompositeController(
                controllers={"pd_controller": pd_controller}, output_clips={"pd_controller": tor_limits}
            )
        return controllers

    def statesType2observation(self, states: StatesType) -> dict:
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
        for name in self.robot_names:
            action[name] = np.full(
                shape=(self.num_envs, self.robots[name].num_dofs), fill_value=np.nan, dtype=np.float32
            )  # It is used to output the final action array (ActionsType) for the backend
            for group_name, map_func in self.action_map.items():
                res = map_func(name=name, action=action, states=self.get_states())
                if res is not None:
                    action[group_name] = res

        return {name: action[name] for name in self.robot_names}

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
        truncated = (self.episode_step >= self.max_episode_steps).item()
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
    def map_cache(self) -> MapCache:
        assert self._map_cache is not None, "Map cache not initialized."
        return self._map_cache

    @property
    def observation_map(self) -> dict[str, Callable]:
        return self.map_cache.observation if self.map_cache.observation is not None else {}

    @property
    def action_map(self) -> dict[str, Callable]:
        return self.map_cache.action if self.map_cache.action is not None else {}

    @property
    def reward_map(self) -> dict[str, Callable]:
        return self.map_cache.reward if self.map_cache.reward is not None else {}

    @property
    def termination_map(self) -> dict[str, Callable]:
        return self.map_cache.termination if self.map_cache.termination is not None else {}

    @property
    def truncation_map(self) -> dict[str, Callable]:
        return self.map_cache.truncation if self.map_cache.truncation is not None else {}

    @property
    def info_map(self) -> dict[str, Any]:
        return self.map_cache.info if self.map_cache.info is not None else {}

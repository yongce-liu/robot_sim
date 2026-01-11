from typing import Callable

import gymnasium as gym
import numpy as np

from robot_sim.configs import ObjectType, SimulatorConfig
from robot_sim.configs.simulator import BackendType
from robot_sim.envs import MapCache, MapEnv
from robot_sim.utils.helper import create_pid_controllers
from robot_sim.utils.math_array import euler_xyz_from_quat


class Twist2Env(MapEnv):
    def __init__(self, config: SimulatorConfig, **kwargs):
        robot_configs = {name: cfg for name, cfg in config.scene.objects.items() if cfg.type == ObjectType.ROBOT}
        assert len(robot_configs) == 1, "Gr00tEnv only supports single robot."
        controllers = (
            None
            if config.backend == BackendType.UNITREE
            else create_pid_controllers(configs=robot_configs, dt=config.sim.dt)
        )

        super().__init__(config=config, controllers=controllers, **kwargs)
        assert len(self.robot_names) == 1 and len(self.robots) == 1, "Twist2Env only supports single robot."
        self.robot_name = self.robot_names[0]
        self.robot = self.robots[self.robot_name]
        self._map_cache: MapCache = self._init_spaces_maps()

    def _init_spaces_maps(self) -> MapCache:
        obs_map = self._init_observation_spaces_map()
        action_map = self._init_action_spaces_map()

        return MapCache(observation=obs_map, action=action_map)

    def _init_observation_spaces_map(self) -> dict[str, Callable]:
        """
        Initialize the observation spaces map for the Gr00t environment.
        Returns:
            A dictionary mapping observation group names to their configurations.
        """
        obs_map: dict[str, Callable] = {}
        _spaces: gym.spaces.Space
        pos_limits = self.robot.get_joint_limits("position", coeff=0.9)
        vel_limits = self.robot.get_joint_limits("velocity", coeff=0.9)
        body_idx = self.robot.get_group_joint_indices(group_name="body", patterns=["^(?!.*hand).*"])  # all except hands
        left_hand_idx = self.robot.get_group_joint_indices(group_name="left_hand")
        right_hand_idx = self.robot.get_group_joint_indices(group_name="right_hand")

        group_name = "body_dof_pos"
        _spaces = gym.spaces.Box(
            low=pos_limits[0][body_idx],
            high=pos_limits[1][body_idx],
            shape=(len(body_idx),),
            dtype=np.float32,
        )
        self.observation_space[group_name] = _spaces
        obs_map[group_name] = (
            lambda states, name=self.robot_name, body_idx=body_idx: states[name].joint_pos[..., body_idx].squeeze()
        )

        group_name = "body_dof_vel"
        _spaces = gym.spaces.Box(
            low=vel_limits[0][body_idx],
            high=vel_limits[1][body_idx],
            shape=(len(body_idx),),
            dtype=np.float32,
        )
        self.observation_space[group_name] = _spaces
        obs_map[group_name] = (
            lambda states, name=self.robot_name, idx=body_idx: states[name].joint_vel[..., idx].squeeze()
        )

        group_name = "rpy"
        _spaces = gym.spaces.Box(
            low=np.array([-np.pi, -np.pi, -np.pi], dtype=np.float32),
            high=np.array([np.pi, np.pi, np.pi], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )
        self.observation_space[group_name] = _spaces
        obs_map[group_name] = lambda states, name=self.robot_name: np.stack(
            euler_xyz_from_quat(states[name].root_state[..., 3:7]), axis=-1
        ).squeeze()

        group_name = "ang_vel"
        _spaces = gym.spaces.Box(
            low=np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.inf, np.inf, np.inf], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )
        self.observation_space[group_name] = _spaces
        obs_map[group_name] = lambda states, name=self.robot_name: states[name].root_state[..., 10:13].squeeze()

        group_name = "left_hand_dof_pos"
        _spaces = gym.spaces.Box(
            low=pos_limits[0][left_hand_idx],
            high=pos_limits[1][left_hand_idx],
            shape=(7,),
            dtype=np.float32,
        )
        self.observation_space[group_name] = _spaces
        obs_map[group_name] = (
            lambda states, name=self.robot_name, idx=left_hand_idx: states[name].joint_pos[..., idx].squeeze()
        )

        group_name = "right_hand_dof_pos"
        _spaces = gym.spaces.Box(
            low=pos_limits[0][right_hand_idx],
            high=pos_limits[1][right_hand_idx],
            shape=(7,),
            dtype=np.float32,
        )
        self.observation_space[group_name] = _spaces
        obs_map[group_name] = (
            lambda states, name=self.robot_name, idx=right_hand_idx: states[name].joint_pos[..., idx].squeeze()
        )

        return obs_map

    def _init_action_spaces_map(self) -> dict[str, Callable]:
        act_map: dict[str, Callable] = {}
        pos_limits = self.robot.get_joint_limits("position", coeff=0.9)

        group_name = "dof_pos"
        self.action_space[group_name] = gym.spaces.Box(
            low=pos_limits[0], high=pos_limits[1], shape=(self.robot.num_dofs,), dtype=np.float32
        )
        act_map[group_name] = lambda action, name=self.robot_name, group_name=group_name, **kwargs: action.update(
            {name: action[group_name]}
        )
        return act_map

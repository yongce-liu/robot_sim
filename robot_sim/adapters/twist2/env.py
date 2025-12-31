from typing import Callable

import gymnasium as gym
import numpy as np

from robot_sim.configs import ControlType, SimulatorConfig
from robot_sim.controllers import CompositeController, PIDController
from robot_sim.envs import MapCache, MapEnv
from robot_sim.utils.math_array import euler_xyz_from_quat


class Twist2Env(MapEnv):
    def __init__(self, config: SimulatorConfig, **kwargs):
        super().__init__(config, **kwargs)

        self._init_specs()
        self._map_cache: MapCache = self._init_spaces_maps()
        self._controller: CompositeController = self._init_controller()

    def _init_spaces_maps(self) -> MapCache:
        obs_map = self._init_observation_spaces_map()
        action_map = self._init_action_spaces_map()

        return MapCache(observation=obs_map, action=action_map)

    def _init_controller(self) -> CompositeController:
        robot_cfg = self.get_object_config(self.robot_name)
        ##### Initialize PD controller
        kp = np.array([joint.stiffness for joint in robot_cfg.joints.values() if joint.actuated], dtype=np.float32)
        kd = np.array([joint.damping for joint in robot_cfg.joints.values() if joint.actuated], dtype=np.float32)
        used_pd_indices = [
            i
            for i, joint in enumerate(robot_cfg.joints.values())
            if ControlType(joint.control_type) == ControlType.TORQUE
        ]
        pd_controller = PIDController(kp=kp, kd=kd, dt=self.step_dt / self.decimation, enabled_indices=used_pd_indices)
        return CompositeController(
            controllers={"pd_controller": pd_controller},
            output_clips={"pd_controller": (-self.torque_limits, self.torque_limits)},
        )

    def _init_specs(self) -> None:
        robot_cfg = self.get_object_config(self.robot_name)
        coeff = robot_cfg.extra.get("soft_coeff", 0.9)
        self.torque_limits = (
            np.array(
                [obj.torque_limit for obj in robot_cfg.joints.values() if obj.actuated],
                dtype=np.float32,
            )
            * coeff
        )
        position_limits = np.array(
            [obj.position_limit for obj in robot_cfg.joints.values() if obj.actuated],
            dtype=np.float32,
        )
        mid = (position_limits[:, 0] + position_limits[:, 1]) / 2
        range_2 = (position_limits[:, 1] - position_limits[:, 0]) * 0.5
        position_limits[:, 0] = mid - coeff * range_2
        position_limits[:, 1] = mid + coeff * range_2
        self.position_limits = position_limits
        self.velocity_limits = np.array(
            [obj.velocity_limit for obj in robot_cfg.joints.values() if obj.actuated],
            dtype=np.float32,
        )
        self.body_idx = np.array([i for i, name in enumerate(robot_cfg.joints.keys()) if "hand" not in name])

    def _init_observation_spaces_map(self) -> dict[str, Callable]:
        """
        Initialize the observation spaces map for the Gr00t environment.
        Returns:
            A dictionary mapping observation group names to their configurations.
        """
        obs_map = {}

        group_name = "body_dof_pos"
        _spaces = gym.spaces.Box(
            low=self.position_limits[self.body_idx, 0],
            high=self.position_limits[self.body_idx, 1],
            shape=(len(self.body_idx),),
            dtype=np.float32,
        )
        self._observation_space_dict[group_name] = _spaces
        obs_map[group_name] = (
            lambda states, name=self.robot_name, body_idx=self.body_idx: states[name].joint_pos[..., body_idx].squeeze()
        )

        group_name = "body_dof_vel"
        _spaces = gym.spaces.Box(
            low=-self.velocity_limits[self.body_idx],
            high=self.velocity_limits[self.body_idx],
            shape=(len(self.body_idx),),
            dtype=np.float32,
        )
        self._observation_space_dict[group_name] = _spaces
        obs_map[group_name] = (
            lambda states, name=self.robot_name, body_idx=self.body_idx: states[name].joint_vel[..., body_idx].squeeze()
        )

        group_name = "rpy"
        _spaces = gym.spaces.Box(
            low=np.array([-np.pi, -np.pi, -np.pi], dtype=np.float32),
            high=np.array([np.pi, np.pi, np.pi], dtype=np.float32),
            shape=(3,),
            dtype=np.float32,
        )
        self._observation_space_dict[group_name] = _spaces
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
        self._observation_space_dict[group_name] = _spaces
        obs_map[group_name] = lambda states, name=self.robot_name: states[name].root_state[..., 10:13].squeeze()

        group_name = "left_hand_dof_pos"

        group_name = "right_hand_dof_pos"

        return obs_map

    def _init_action_spaces_map(self) -> dict[str, Callable]:
        self._action_space_dict = {}
        act_map = {}
        group_name = "dof_pos"
        _spaces = gym.spaces.Box(
            low=self.position_limits[:, 0],
            high=self.position_limits[:, 1],
            shape=(self.num_dofs[self.robot_name],),
            dtype=np.float32,
        )
        self._action_space_dict[group_name] = _spaces
        act_map[group_name] = lambda action, name=self.robot_name, group_name=group_name, **kwargs: action.update(
            {name: action[group_name]}
        )
        return act_map

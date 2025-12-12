from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from loguru import logger as log

from robot_sim.configs import ObjectConfig, PhysicsConfig, RobotConfig, SimulatorConfig, TerrainConfig

########################## Data Structures ##########################

ArrayTypes = torch.Tensor | np.ndarray
ActionType = dict[str, ArrayTypes]


@dataclass
class ObjectState:
    """State of a single object."""

    root_state: ArrayTypes
    """Root state ``[pos, quat, lin_vel, ang_vel]``. Shape is (num_envs, 13)."""
    body_names: list[str] | None = None
    """Body names. This is only available for articulation objects."""
    body_state: ArrayTypes | None = None
    """Body state ``[pos, quat, lin_vel, ang_vel]``. Shape is (num_envs, num_bodies, 13). This is only available for articulation objects."""
    joint_pos: ArrayTypes | None = None
    """Joint positions. Shape is (num_envs, num_joints). This is only available for articulation objects."""
    joint_vel: ArrayTypes | None = None
    """Joint velocities. Shape is (num_envs, num_joints). This is only available for articulation objects."""


@dataclass
class RobotState:
    """State of a single robot."""

    root_state: ArrayTypes
    """Root state ``[pos, quat, lin_vel, ang_vel]``. Shape is (num_envs, 13)."""
    body_names: list[str]
    """Body names."""
    body_state: ArrayTypes
    """Body state ``[pos, quat, lin_vel, ang_vel]``. Shape is (num_envs, num_bodies, 13)."""
    joint_pos: ArrayTypes
    """Joint positions. Shape is (num_envs, num_joints)."""
    joint_vel: ArrayTypes
    """Joint velocities. Shape is (num_envs, num_joints)."""
    joint_pos_target: ArrayTypes
    """Joint positions target. Shape is (num_envs, num_joints)."""
    joint_vel_target: ArrayTypes
    """Joint velocities target. Shape is (num_envs, num_joints)."""
    joint_effort_target: ArrayTypes
    """Joint effort targets. Shape is (num_envs, num_joints)."""
    sensors: dict[str, ArrayTypes] = field(default_factory=dict)
    """Sensor readings. Each sensor has shape (num_envs, sensor_dim)."""
    extras: dict = field(default_factory=dict)
    """Extra information."""


@dataclass
class ArrayState:
    """A dictionary that holds the states of all robots and objects in tensor format.

    The keys are the names of the robots and objects, and the values are tensors representing their states.
    The tensor shape is (num_envs, state_dim), where num_envs is the number of environments, and state_dim is the dimension of the state for each robot or object.
    """

    objects: dict[str, ObjectState]
    robots: dict[str, RobotState]
    extras: dict = field(default_factory=dict)


########################### Base Backend ##########################
class BaseBackend(ABC):
    """Base class for simulation handler."""

    def __init__(self, config: SimulatorConfig, optional_queries: dict[str, Any] | None = None):
        self.optional_queries = optional_queries

        ## For quick reference
        self.config: SimulatorConfig = config
        self.cfg_phyx: PhysicsConfig = config.sim
        if config.scene.path is None:
            self.robots: dict[str, RobotConfig] = config.scene.robots
            self.objects: dict[str, ObjectConfig] = config.scene.objects
            self.terrain: TerrainConfig = config.scene.terrain
        else:
            log.info("Scene file provided, ignoring robots, objects, and terrain from config.")
            self.robots = {}
            self.objects = {}
            self.terrain = None
        # TODO: maybe need to add more objects like terrains, lights, cameras, etc.

        self._state_cache_expire = True
        self._states: ArrayState = None

        # Constant tensors
        self._full_env_ids = (
            np.arange(self.num_envs) if self.device == "cpu" else torch.arange(self.num_envs, device=self.device)
        )
        self._joint_indices: dict[str, list[str]] = defaultdict(list)  # robot/object name -> list[joint name]
        self._body_indices: dict[str, list[str]] = defaultdict(list)  # robot/object name -> list[body name]

    def launch(self) -> None:
        """Launch the simulation."""
        self._launch()
        if self.optional_queries is None:
            self.optional_queries = {}
        for query_name, query_type in self.optional_queries.items():
            query_type.bind_handler(self)

    def simulate(self):
        """Simulate the environment."""
        self._state_cache_expire = True
        self._simulate()

    def set_states(self, states: ArrayState, env_ids: ArrayTypes | None = None) -> None:
        """Set the states of the environment."""
        self._state_cache_expire = True
        self._set_states(states, env_ids)

    def set_actions(self, actions: ActionType) -> None:
        """Set the dof targets of the robot.

        Args:
            obj_name (str): The name of the robot
            actions (dict[str, ActionType]): The target actions for the robot
        """
        self._set_actions(actions)

    def get_states(self, env_ids: ArrayTypes | None = None) -> ArrayState:
        """Get the states of the environment."""
        if self._state_cache_expire:
            self._states = self._get_states(env_ids=env_ids)
            self._state_cache_expire = False
        return self._states

    def get_extra(self):
        """Get the extra information of the environment."""
        ret_dict = {}
        for query_name, query_type in self.optional_queries.items():
            ret_dict[query_name] = query_type()
        return ret_dict

    # Abstract Methods
    @abstractmethod
    def _launch(self) -> None:
        """Launch the simulation.
        For a new simulator, you should implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def render(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        """Close the simulation."""
        raise NotImplementedError

    @abstractmethod
    def _simulate(self):
        """Simulate the environment for one time step.
        For a new simulator, you should implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def _set_states(self, states: ArrayState, env_ids: ArrayTypes | None = None) -> None:
        """Set the states of the environment.
        For a new simulator, you should implement this method.

        Args:
            states (dict): A dictionary containing the states of the environment
            env_ids (list[int]): List of environment ids to set the states. If None, set the states of all environments
        """
        raise NotImplementedError

    @abstractmethod
    def _set_actions(self, actions: ActionType) -> None:
        """Set the dof targets of the environment.
        For a new simulator, you should implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_states(self, env_ids: ArrayTypes | None = None) -> ArrayState:
        """Get the states of the environment.
        For a new simulator, you should implement this method.

        Args:
            env_ids: List of environment ids to get the states from. If None, get the states of all environments.

        Returns:
            dict: A dictionary containing the states of the environment
        """
        raise NotImplementedError

    # Properties
    @property
    def num_envs(self) -> int:
        return self.cfg_phyx.num_envs

    @property
    def headless(self) -> bool:
        return self.cfg_phyx.headless

    @property
    def device(self) -> str:
        return self.cfg_phyx.device

    @property
    def full_env_ids(self) -> torch.Tensor:
        """Get all environment ids."""
        return self._full_env_ids

    @property
    def joint_indices(self) -> dict[str, dict[str, int]]:
        """Get the joint indices of all robots and objects."""
        return self._joint_indices

    @property
    def body_indices(self) -> dict[str, dict[str, int]]:
        """Get the body indices of all robots and objects."""
        return self._body_indices

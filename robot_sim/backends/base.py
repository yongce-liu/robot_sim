from abc import ABC, abstractmethod
from argparse import Action
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from loguru import logger as log

from robot_sim.configs import ObjectConfig, PhysicsConfig, RobotConfig, SimulatorConfig

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
        self.cfg_phyx: PhysicsConfig = config.sim
        self.robots: dict[str, RobotConfig] = config.scene.robots
        self.objects: dict[str, ObjectConfig] = config.scene.objects

        self._state_cache_expire = True
        self._states: ArrayState = None

    def launch(self) -> None:
        """Launch the simulation."""
        if self.optional_queries is None:
            self.optional_queries = {}
        for query_name, query_type in self.optional_queries.items():
            query_type.bind_handler(self)
        # raise NotImplementedError

    def render(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        """Close the simulation."""
        raise NotImplementedError

    ############################################################
    ## Set states
    ############################################################
    @abstractmethod
    def _set_states(self, states: ArrayState, env_ids: list[int] | None = None) -> None:
        """Set the states of the environment.
        For a new simulator, you should implement this method.

        Args:
            states (dict): A dictionary containing the states of the environment
            env_ids (list[int]): List of environment ids to set the states. If None, set the states of all environments
        """
        raise NotImplementedError

    def set_states(self, states: ArrayState, env_ids: list[int] | None = None) -> None:
        """Set the states of the environment."""
        self._state_cache_expire = True
        self._set_states(states, env_ids)

    # @abstractmethod
    def _set_dof_targets(self, actions: ActionType) -> None:
        """Set the dof targets of the environment.
        For a new simulator, you should implement this method.
        """
        raise NotImplementedError

    def set_dof_targets(self, actions: ActionType) -> None:
        """Set the dof targets of the robot.

        Args:
            obj_name (str): The name of the robot
            actions (list[Action]): The target actions for the robot
        """
        self._set_dof_targets(actions)

    ############################################################
    ## Get states
    ############################################################
    @abstractmethod
    def _get_states(self, env_ids: list[int] | None = None) -> ArrayState:
        """Get the states of the environment.
        For a new simulator, you should implement this method.

        Args:
            env_ids: List of environment ids to get the states from. If None, get the states of all environments.

        Returns:
            dict: A dictionary containing the states of the environment
        """
        raise NotImplementedError

    def get_states(self, env_ids: list[int] | None = None) -> ArrayState:
        """Get the states of the environment."""
        if self._state_cache_expire:
            self._states = self._get_states(env_ids=env_ids)
            self._state_cache_expire = False
        return self._states

    ############################################################
    ## Get extra queries
    ############################################################
    def get_extra(self):
        """Get the extra information of the environment."""
        ret_dict = {}
        for query_name, query_type in self.optional_queries.items():
            ret_dict[query_name] = query_type()
        return ret_dict

    ############################################################
    ## Simulate
    ############################################################
    @abstractmethod
    def _simulate(self):
        """Simulate the environment for one time step.
        For a new simulator, you should implement this method.
        """
        raise NotImplementedError

    def simulate(self):
        """Simulate the environment."""
        self._state_cache_expire = True
        self._simulate()

    ############################################################
    ## Misc
    ############################################################
    # @abstractmethod
    def _get_joint_names(self, obj_name: str, sort: bool = True) -> list[str]:
        """Get the joint names for a given object.
        For a new simulator, you should implement this method.

        Note:
            Different simulators may have different joint order, but joint names should be the same.

        Args:
            obj_name (str): The name of the object.
            sort (bool): Whether to sort the joint names. Default is True. If True, the joint names are returned in alphabetical order. If False, the joint names are returned in the order defined by the simulator.

        Returns:
            list[str]: A list of joint names. For non-articulation objects, return an empty list.
        """
        raise NotImplementedError

    def get_joint_names(self, obj_name: str, sort: bool = True) -> list[str]:
        """Get the joint names for a given object."""
        return self._get_joint_names(obj_name, sort)

    def get_joint_reindex(self, obj_name: str, inverse: bool = False) -> list[int]:
        """Get the reindexing order for joint indices of a given object. The returned indices can be used to reorder the joints such that they are sorted alphabetically by their names.

        Args:
            obj_name (str): The name of the object.
            inverse (bool): Whether to return the inverse reindexing order. Default is False.

        Returns:
            list[int]: A list of joint indices that specifies the order to sort the joints alphabetically by their names.
               The length of the list matches the number of joints. If ``inverse`` is True, the returned list is inversed, which means they can be used to restore the original order.

        Example:
            Suppose ``obj_name = "h1"``, and the ``h1`` has joints:

            index 0: ``"hip"``

            index 1: ``"knee"``

            index 2: ``"ankle"``

            This function will return: ``[2, 0, 1]``, which corresponds to the alphabetical order:
                ``"ankle"``, ``"hip"``, ``"knee"``.
        """
        if not hasattr(self, "_joint_reindex_cache"):
            self._joint_reindex_cache = {}
            self._joint_reindex_cache_inverse = {}

        if obj_name not in self._joint_reindex_cache:
            origin_joint_names = self._get_joint_names(obj_name, sort=False)
            sorted_joint_names = self._get_joint_names(obj_name, sort=True)
            self._joint_reindex_cache[obj_name] = [origin_joint_names.index(jn) for jn in sorted_joint_names]
            self._joint_reindex_cache_inverse[obj_name] = [sorted_joint_names.index(jn) for jn in origin_joint_names]

        return self._joint_reindex_cache_inverse[obj_name] if inverse else self._joint_reindex_cache[obj_name]

    # @abstractmethod
    def _get_body_names(self, obj_name: str, sort: bool = True) -> list[str]:
        """Get the body names for a given object.
        For a new simulator, you should implement this method.

        Note:
            Different simulators may have different body order, but body names should be the same.

        Args:
            obj_name (str): The name of the object.
            sort (bool): Whether to sort the body names. Default is True. If True, the body names are returned in alphabetical order. If False, the body names are returned in the order defined by the simulator.

        Returns:
            list[str]: A list of body names. For non-articulation objects, return an empty list.
        """
        raise NotImplementedError

    def get_body_names(self, obj_name: str, sort: bool = True) -> list[str]:
        """Get the body names for a given object."""
        return self._get_body_names(obj_name, sort)

    def get_body_reindex(self, obj_name: str) -> list[int]:
        """Get the reindexing order for body indices of a given object. The returned indices can be used to reorder the bodies such that they are sorted alphabetically by their names.

        Args:
            obj_name (str): The name of the object.

        Returns:
            list[int]: A list of body indices that specifies the order to sort the bodies alphabetically by their names.
               The length of the list matches the number of bodies.

        Example:
            Suppose ``obj_name = "h1"``, and the ``h1`` has the following bodies:

                - index 0: ``"torso"``
                - index 1: ``"left_leg"``
                - index 2: ``"right_leg"``

            This function will return: ``[1, 2, 0]``, which corresponds to the alphabetical order:
                ``"left_leg"``, ``"right_leg"``, ``"torso"``.
        """
        if not hasattr(self, "_body_reindex_cache"):
            self._body_reindex_cache = {}

        if obj_name not in self._body_reindex_cache:
            origin_body_names = self._get_body_names(obj_name, sort=False)
            sorted_body_names = self._get_body_names(obj_name, sort=True)
            self._body_reindex_cache[obj_name] = [origin_body_names.index(bn) for bn in sorted_body_names]

        return self._body_reindex_cache[obj_name]

    ############################################################
    ## GS Renderer
    ############################################################
    def _get_camera_params(self, camera):
        """Get the camera parameters for GS rendering.
        For a new simulator, you should implement this method.
        Args:
            camera: PinholeCameraCfg object

        Returns:
            Ks: (3, 3) intrinsic matrix
            c2w: (4, 4) camera-to-world transformation matrix
        """
        raise NotImplementedError

    @property
    def num_envs(self) -> int:
        return self.cfg_phyx.num_envs

    @property
    def headless(self) -> bool:
        return self.cfg_phyx.headless

    @property
    def device(self) -> str:
        return self.cfg_phyx.device

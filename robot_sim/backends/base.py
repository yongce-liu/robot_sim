from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Any

import numpy as np
import torch

from robot_sim.backends.types import ActionType, ArrayState, ArrayTypes, Buffer
from robot_sim.configs import (
    BackendType,
    ObjectConfig,
    PhysicsConfig,
    SimulatorConfig,
    TerrainConfig,
)


########################### Base Backend ##########################
class BaseBackend(ABC):
    """Base class for simulation handler."""

    def __init__(self, config: SimulatorConfig, optional_queries: dict[str, Any] | None = None):
        self.optional_queries = optional_queries if optional_queries is not None else {}

        ## For quick reference
        self.config: SimulatorConfig = config
        self.type: BackendType = config.backend
        self.cfg_phyx: PhysicsConfig = config.sim
        self.objects: dict[str, ObjectConfig] = config.scene.objects  # robots + objects
        self.terrain: TerrainConfig | None = config.scene.terrain
        """
        if config.scene.path is None:
            self.objects: dict[str, ObjectConfig] = config.scene.objects  # robots + objects
            self.terrain: TerrainConfig | None = config.scene.terrain
        else:
            logger.info("Scene file provided, ignoring robots, objects, and terrain from config.")
            self.objects = {}
            self.terrain = None
        """
        # TODO: maybe need to add more objects like terrains, lights, cameras, etc.

        self._state_cache_expire = True
        self._states: ArrayState = None

        # Constants
        self.is_launched = False
        self._sim_cnt = 0
        self._sim_freq = int(1.0 / self.cfg_phyx.dt)
        self._full_env_ids = (
            np.arange(self.num_envs) if self.device == "cpu" else torch.arange(self.num_envs, device=self.device)
        )
        # you can use it to store anything, e.g., default joint names order/ body names order, etc.
        self._buffer_dict = defaultdict(Buffer)

    def _init_backend(self, *args, **kwargs) -> None:
        """Initialize the backend simulator.
        You can do some preparation work here before launching the simulator.
        Or you can directly put the initialization code in the `_launch` method.
        You also don't overide this method if you don't need it.
        """
        pass

    def launch(self) -> None:
        """Launch the simulation."""
        self._sim_cnt = 0
        self._init_backend()
        self._bind_sensors_queries()
        self._launch()
        self._refresh_sensors(self._sim_cnt)
        self.is_launched = True

    def _bind_sensors_queries(self) -> None:
        """Bind sensors to the backend."""
        for obj_name, obj_buffer in self._buffer_dict.items():
            for sensor_name, sensor_instance in obj_buffer.sensors.items():
                sensor_instance.bind(self, obj_name, sensor_name)
        for query_name, query_type in self.optional_queries.items():
            query_type.bind(self)

    def render(self) -> None:
        if self._sim_cnt % self.cfg_phyx.render_interval == 0 and not self.headless:
            self._render()

    # def get_world_image(self):
    #     """Get the world image from the backend."""
    #     raise NotImplementedError("get_world_image() is not implemented for this backend.")

    def simulate(self):
        """Simulate the environment."""
        self._sim_cnt = (self._sim_cnt + 1) % self._sim_freq
        self._state_cache_expire = True
        self._simulate()
        self._refresh_sensors(self._sim_cnt)
        self.render()

    def _refresh_sensors(self, cnt: int) -> None:
        for sensor_dict in self._buffer_dict.values():
            for sensor in sensor_dict.sensors.values():
                sensor(cnt)

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

    def get_states(self) -> ArrayState:
        """Get the states of the environment.
        It will return all env state
        """
        if self._state_cache_expire:
            self._states = self._get_states()
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
    def _render(self) -> None:
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
    def _get_states(self) -> ArrayState:
        """Get the states of the environment.
        For a new simulator, you should implement this method.

        Args:
            env_ids: List of environment ids to get the states from. If None, get the states of all environments.

        Returns:
            dict: A dictionary containing the states of the environment
        """
        raise NotImplementedError

    @abstractmethod
    def _update_buffer_dict(self, *args, **kwargs) -> None:
        """Update the buffer dict with the model and config information.
        For a new simulator, you should implement this method.
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

    # Utility functions for buffers
    # public functions
    def get_joint_names(self, name: str) -> list[str]:
        """Get the joint indices of all robots and objects."""
        return self._buffer_dict[name].joint_names

    def get_body_names(self, name: str) -> list[str]:
        """Get the body indices of all robots and objects."""
        return self._buffer_dict[name].body_names

    def get_actuator_names(self, name: str) -> list[str]:
        """Get the actuator names of all robots and objects."""
        return self._buffer_dict[name].actuator_names

    # private functions
    # def get_sensors(self, name: str) -> dict[str, SensorConfig]:
    #     """Get the sensor configs of all robots and objects."""
    #     return self._buffer_dict[name].sensors

    def _get_joint_indices(self, name: str) -> list[int]:
        """Get the joint indices of all robots and objects."""
        return self._buffer_dict[name].joint_indices

    def _get_action_indices(self, name: str) -> list[int]:
        """Get the action indices of all robots and objects."""
        return self._buffer_dict[name].action_indices

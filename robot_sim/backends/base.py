from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from loguru import logger

from robot_sim.backends.sensors import _SENSOR_TYPE_REGISTRY
from robot_sim.backends.types import ActionsType, ArrayType, Buffer, StatesType
from robot_sim.configs import BackendType, ObjectConfig, PhysicsConfig, SimulatorConfig, TerrainConfig, VisualConfig


########################### Base Backend ##########################
class BaseBackend(ABC):
    """Base class for simulation handler."""

    def __init__(self, config: SimulatorConfig, optional_queries: dict[str, Any] | None = None):
        self.optional_queries = optional_queries if optional_queries is not None else {}
        self.type: BackendType = config.backend

        ## For quick reference
        self._config: SimulatorConfig = config
        # self._cfg_phyx: PhysicsConfig = config.sim
        # self._objects: dict[str, ObjectConfig] = config.scene.objects  # robots + objects
        # self._terrain: TerrainConfig | None = config.scene.terrain
        """
        if config.scene.path is None:
            self._objects: dict[str, ObjectConfig] = config.scene.objects  # robots + objects
            self._terrain: TerrainConfig | None = config.scene.terrain
        else:
            logger.info("Scene file provided, ignoring robots, objects, and terrain from config.")
            self._objects = {}
            self._terrain = None
        """
        # TODO: maybe need to add more objects like terrains, lights, cameras, etc.

        self._state_cache_expire = True
        self._states: StatesType | None = None

        # Constants
        self.is_launched = False
        self._sim_cnt = 0
        self._sim_freq = int(1.0 / self.sim_config.dt)
        self._full_env_ids = (
            np.arange(self.num_envs) if self.device == "cpu" else torch.arange(self.num_envs, device=self.device)
        )
        # you can use it to store anything, e.g., default joint names order/ body names order, etc.
        self._buffer_dict: dict[str, Buffer] = defaultdict(Buffer)
        self.__cache = {}

    def launch(self) -> None:
        """Launch the simulation."""
        self._sim_cnt = 0
        self._init_buffer_dict()
        self._bind_sensors_queries()
        self._launch()
        self._refresh_sensors(self._sim_cnt)
        self.set_states(self.initial_states)
        self._render()
        self.is_launched = True

    def _bind_sensors_queries(self) -> None:
        """Bind sensors to the backend."""
        for obj_name, obj_buffer in self._buffer_dict.items():
            for sensor_name, sensor_instance in obj_buffer.sensors.items():
                sensor_instance.bind(self, obj_name, sensor_name)
        for query_name, query_type in self.optional_queries.items():
            query_type.bind(self)

    def render(self) -> None:
        if self._sim_cnt % self.sim_config.render_interval == 0 and not self.headless:
            self._render()

    # def get_world_image(self):
    #     """Get the world image from the backend."""
    #     raise NotImplementedError("get_world_image() is not implemented for this backend.")

    def reset(self, states: StatesType | None = None) -> None:
        """Reset the environment to the initial states or given states."""
        self._state_cache_expire = True
        self._sim_cnt = 0
        if states is None:
            self.set_states(self.initial_states)
        else:
            self.set_states(states)
        self._refresh_sensors(self._sim_cnt)
        self.render()

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

    def set_states(self, states: StatesType, env_ids: ArrayType | None = None) -> None:
        """Set the states of the environment."""
        self._state_cache_expire = True
        self._set_states(states, env_ids)

    def set_actions(self, actions: ActionsType) -> None:
        """Set the dof targets of the robot.

        Args:
            obj_name (str): The name of the robot
            actions (dict[str, ActionsType]): The target actions for the robot
        """
        self._set_actions(actions)

    def get_states(self) -> StatesType:
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

    def _init_buffer_dict(self) -> None:
        """Initialize the buffer dict with the model and config information.
        For a new simulator, you should implement _update_buffer_dict.
        """
        for obj_name, obj_cfg in self.objects.items():
            # joint names
            self._buffer_dict[obj_name].actuator_names = (
                [k for k, v in obj_cfg.joints.items() if v.actuated] if obj_cfg.joints else None
            )
            self._buffer_dict[obj_name].joint_names = list(obj_cfg.joints.keys()) if obj_cfg.joints else None
            self._buffer_dict[obj_name].body_names = list(obj_cfg.bodies.keys()) if obj_cfg.bodies else None

            for sensor_name, sensor_cfg in obj_cfg.sensors.items():
                sensor_type = sensor_cfg.type
                if sensor_type in _SENSOR_TYPE_REGISTRY:
                    sensor_instance = _SENSOR_TYPE_REGISTRY[sensor_type](sensor_cfg)
                    # sensor_instance.bind(self, obj_name, sensor_name)
                    self._buffer_dict[obj_name].sensors[sensor_name] = sensor_instance
                else:
                    logger.error(
                        f"Unsupported sensor type '{sensor_type}' for sensor '{sensor_name}' in object '{obj_name}'"
                    )
            # self._buffer_dict[obj_name].config = obj_cfg
        self._update_buffer_indices()

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
    def _set_states(self, states: StatesType, env_ids: ArrayType | None = None) -> None:
        """Set the states of the environment.
        For a new simulator, you should implement this method.

        Args:
            states (dict): A dictionary containing the states of the environment
            env_ids (list[int]): List of environment ids to set the states. If None, set the states of all environments
        """
        raise NotImplementedError

    @abstractmethod
    def _set_actions(self, actions: ActionsType) -> None:
        """Set the dof targets of the environment.
        For a new simulator, you should implement this method.
        """
        raise NotImplementedError

    @abstractmethod
    def _get_states(self) -> StatesType:
        """Get the states of the environment.
        For a new simulator, you should implement this method.

        Args:
            env_ids: List of environment ids to get the states from. If None, get the states of all environments.

        Returns:
            dict: A dictionary containing the states of the environment
        """
        raise NotImplementedError

    @abstractmethod
    def _update_buffer_indices(self) -> None:
        """Update the buffer dict with the model and config information.
        For a new simulator, you should implement this method.
        """
        raise NotImplementedError

    # Properties
    @property
    def num_envs(self) -> int:
        return self.sim_config.num_envs

    @property
    def headless(self) -> bool:
        return self.sim_config.headless

    @property
    def device(self) -> str:
        return self.sim_config.device

    @property
    def full_env_ids(self) -> torch.Tensor:
        """Get all environment ids."""
        return self._full_env_ids

    @property
    def config(self) -> SimulatorConfig:
        """Get the simulator configuration."""
        return self._config

    @property
    def objects(self) -> dict[str, ObjectConfig]:
        """Get the object configurations."""
        return self._config.scene.objects

    @property
    def terrain(self) -> TerrainConfig | None:
        """Get the terrain configuration."""
        return self._config.scene.terrain

    @property
    def visual(self) -> VisualConfig | None:
        """Get whether visualization is enabled."""
        return self._config.scene.visual

    @property
    def sim_config(self) -> PhysicsConfig:
        """Get the physics configuration."""
        return self._config.sim

    @property
    def initial_states(self) -> StatesType:
        """Get the initial states of the environment."""
        if "_initial_states" not in self.__cache:
            template = self.get_states()
            for obj_name, obj_cfg in self.objects.items():
                root_state = np.concatenate([np.array(obj_cfg.pose), np.array(obj_cfg.twist)], axis=-1)
                i_j_pos = [joint.default_position for joint in obj_cfg.joints.values()] if obj_cfg.joints else []
                if isinstance(template[obj_name].root_state, np.ndarray):
                    template[obj_name].root_state = root_state[None, ...].repeat(self.num_envs, axis=0)
                    template[obj_name].joint_pos = np.array(i_j_pos)[None, ...].repeat(self.num_envs, axis=0)
                else:
                    template[obj_name].root_state = torch.tensor(root_state, device=self.device)[None, ...].repeat(
                        self.num_envs
                    )
                    template[obj_name].joint_pos[..., 0] = torch.tensor(i_j_pos, device=self.device)[None, ...].repeat(
                        self.num_envs
                    )
            self.__cache["_initial_states"] = deepcopy(template)
        return self.__cache["_initial_states"]

    # Utility functions for buffers
    # public functions
    def get_joint_names(self, name: str, prefix: str = "") -> list[str]:
        """Get the joint indices of all robots and objects."""
        idx = f"joint_names/{name}/{prefix}"
        if idx not in self.__cache:
            self.__cache[idx] = [prefix + jn for jn in self._buffer_dict[name].joint_names]
        return self.__cache[idx]

    def get_body_names(self, name: str, prefix: str = "") -> list[str]:
        """Get the body indices of all robots and objects."""
        idx = f"body_names/{name}/{prefix}"
        if idx not in self.__cache:
            self.__cache[idx] = [prefix + bn for bn in self._buffer_dict[name].body_names]
        return self.__cache[idx]

    def get_actuator_names(self, name: str, prefix: str = "") -> list[str]:
        """Get the actuator names of all robots and objects."""
        idx = f"actuator_names/{name}/{prefix}"
        if idx not in self.__cache:
            self.__cache[idx] = [prefix + an for an in self._buffer_dict[name].actuator_names]
        return self.__cache[idx]

    # private functions
    # def get_sensors(self, name: str) -> dict[str, SensorConfig]:
    #     """Get the sensor configs of all robots and objects."""
    #     return self._buffer_dict[name].sensors

    def _get_joint_indices_map(self, name: str, reverse: bool = False) -> list[int]:
        """Get the joint indices map between backend and config order of all robots and objects."""
        if reverse:
            return self._buffer_dict[name].joint_indices_reverse
        return self._buffer_dict[name].joint_indices

    def _get_body_indices_map(self, name: str, reverse: bool = False) -> list[int]:
        """Get the body indices map between backend and config order of all robots and objects."""
        if reverse:
            return self._buffer_dict[name].body_indices_reverse
        return self._buffer_dict[name].body_indices

    def _get_action_indices_map(self, name: str, reverse: bool = False) -> list[int]:
        """Get the action indices map between backend and config order of all robots and objects."""
        if reverse:
            return self._buffer_dict[name].action_indices_reverse
        return self._buffer_dict[name].action_indices

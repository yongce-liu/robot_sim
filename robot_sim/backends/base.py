from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from typing import Any, cast

import numpy as np
import torch
from loguru import logger

from robot_sim.backends.sensors import _SENSOR_TYPE_REGISTRY, BaseSensor
from robot_sim.configs import (
    BackendType,
    ObjectConfig,
    PhysicsConfig,
    RobotModel,
    SimulatorConfig,
    TerrainConfig,
    VisualConfig,
)
from robot_sim.configs.types import ActionsType, ArrayType, StatesType
from robot_sim.controllers import CompositeController
from robot_sim.utils.helper import wrap_array


########################### Base Backend ##########################
class BaseBackend(ABC):
    """Base class for simulation handler."""

    def __init__(
        self,
        config: SimulatorConfig,
        controllers: dict[str, CompositeController] | None = None,
        optional_queries: dict[str, Any] | None = None,
    ):
        ## For quick reference
        self._config: SimulatorConfig = config
        # low-level controller duration=sim.dt, in general, pid is used
        self._controllers = controllers if controllers is not None else {}
        self._optional_queries = optional_queries if optional_queries is not None else {}
        self._full_env_ids: ArrayType = wrap_array(np.arange(self.num_envs), self.device)
        # obj_name -> sensor_name -> sensor_instance
        self._sensors: dict[str, dict[str, BaseSensor]] = defaultdict(dict)
        # you can use it to store anything, e.g., default joint names order/ body names order, etc.
        self._cache: dict[str, Any] = {}
        # Constants
        self._is_launched = False
        self._sim_cnt = 0
        # State Cache Flag
        self._state_cache_expire = True

    def launch(self) -> None:
        """Launch the simulation."""
        self._sim_cnt = 0
        self._bind_sensors_queries()
        self._launch()
        self._refresh_sensors(self.sim_cnt)
        self.set_states(self.initial_states)
        self._render()
        self._is_launched = True

    def _bind_sensors_queries(self) -> None:
        """Bind sensors to the backend."""
        for obj_name, obj_cfg in self.cfg_objects.items():
            for sensor_name, sensor_cfg in obj_cfg.sensors.items():
                sensor_type = sensor_cfg.type
                if sensor_type in _SENSOR_TYPE_REGISTRY:
                    sensor_instance = _SENSOR_TYPE_REGISTRY[sensor_type](sensor_cfg)
                    sensor_instance.bind(self, obj_name, sensor_name)
                    self._sensors[obj_name][sensor_name] = sensor_instance
                else:
                    logger.error(
                        f"Unsupported sensor type '{sensor_type}' for sensor '{sensor_name}' in object '{obj_name}'"
                    )
        for name, query_impl in self._optional_queries.items():
            query_impl.bind(self)

    def render(self) -> None:
        if self.sim_cnt % self.render_interval == 0 and not self.headless:
            self._render()

    def reset(self, states: StatesType | None = None, env_ids: ArrayType | None = None) -> None:
        """Reset the environment to the initial states or given states."""
        self._state_cache_expire = True
        self._sim_cnt = 0
        states = states if states is not None else self.initial_states
        self.set_states(states=states, env_ids=env_ids)
        self._refresh_sensors(self.sim_cnt)
        self.render()

    def simulate(self):
        """Simulate the environment."""
        self._sim_cnt = (self.sim_cnt + 1) % self.sim_freq
        self._state_cache_expire = True
        self._simulate()
        self._refresh_sensors(self.sim_cnt)
        self.render()

    def _refresh_sensors(self, cnt: int) -> None:
        for name, sub_sensors in self.sensors.items():
            for sensor_name, sensor in sub_sensors.items():
                sensor(cnt)

    def get_queries(self):
        """Get the extra information of the environment."""
        ret_dict = {}
        for query_name, query_impl in self._optional_queries.items():
            ret_dict[query_name] = query_impl()
        return ret_dict

    def set_states(self, states: StatesType, env_ids: ArrayType | None = None) -> None:
        """Set the states of the environment."""
        if env_ids is None:
            env_ids = self._full_env_ids
        self._state_cache_expire = True
        self._set_states(states=states, env_ids=env_ids)

    def set_actions(self, actions: ActionsType, env_ids: ArrayType | None = None) -> None:
        """Set the dof targets of the robot.

        Args:
            obj_name (str): The name of the robot
            actions (dict[str, ActionsType]): The target actions for the robot
        """
        if env_ids is None:
            env_ids = self._full_env_ids
        outputs: ActionsType = dict(actions)
        for name, controller in self._controllers.items():
            states = self.get_states()
            outputs[name] = controller.compute(state=states[name], target=actions[name])
        self._set_actions(actions=outputs, env_ids=env_ids)

    def get_states(self) -> StatesType:
        """Get the states of the environment.
        It will return all env state
        """
        if self._state_cache_expire:
            self._states = self._get_states()
            self._state_cache_expire = False
        return self._states

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

    def get_rgb_image(self) -> np.ndarray | None:
        """Get the RGB image of the environment.

        Returns:
            np.ndarray: The RGB image of the environment.
        """
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
    def _set_states(self, states: StatesType, env_ids: ArrayType) -> None:
        """Set the states of the environment.
        For a new simulator, you should implement this method.

        Args:
            states (dict): A dictionary containing the states of the environment
            env_ids (list[int]): List of environment ids to set the states. If None, set the states of all environments
        """
        raise NotImplementedError

    @abstractmethod
    def _set_actions(self, actions: ActionsType, env_ids: ArrayType) -> None:
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

    # Config Properties
    @property
    def cfg_objects(self) -> dict[str, ObjectConfig]:
        """Get the object configurations."""
        return self._config.scene.objects

    @property
    def cfg_terrain(self) -> TerrainConfig | None:
        """Get the terrain configuration."""
        return self._config.scene.terrain

    @property
    def cfg_visual(self) -> VisualConfig | None:
        """Get whether visualization is enabled."""
        return self._config.scene.visual

    @property
    def cfg_sim(self) -> PhysicsConfig:
        """Get the physics configuration."""
        return self._config.sim

    @property
    def cfg_spec(self) -> dict:
        return self._config.spec.get(self.type.value, {})

    @property
    def cfg_extras(self) -> dict:
        return self._config.extras

    @property
    def type(self) -> BackendType:
        """return the backend type"""
        return self._config.backend

    # Quick ref config properties
    @property
    def num_envs(self) -> int:
        """Get the number of environments."""
        return self.cfg_sim.num_envs

    @property
    def device(self) -> str:
        """Get the device of the simulation."""
        return self.cfg_sim.device

    @property
    def headless(self) -> bool:
        """Get whether the simulation is headless."""
        return self.cfg_sim.headless

    @property
    def render_interval(self) -> int:
        """Get the render interval of the simulation."""
        return self.cfg_sim.render_interval

    @property
    def sensors(self) -> dict[str, dict[str, BaseSensor]]:
        """Get the sensors in the environment."""
        return self._sensors

    @property
    def is_launched(self) -> bool:
        """Get whether the simulation is launched."""
        return self._is_launched

    @property
    def sim_cnt(self) -> int:
        """Get the current simulation count."""
        return self._sim_cnt

    @property
    def sim_freq(self) -> int:
        """Get the simulation frequency."""
        return int(1.0 / self.cfg_sim.dt)

    # other properties
    @property
    def initial_states(self) -> StatesType:
        """Get the initial states of the environment."""
        if "_initial_states" not in self._cache:
            template = self.get_states()
            for obj_name, obj_cfg in self.cfg_objects.items():
                root_state = np.concatenate([np.array(obj_cfg.pose), np.array(obj_cfg.twist)], axis=-1)
                i_j_pos = [joint.default_position for joint in obj_cfg.joints.values()] if obj_cfg.joints else []
                if isinstance(template[obj_name].root_state, np.ndarray):
                    template[obj_name].root_state = root_state[None, ...].repeat(self.num_envs, axis=0)
                    template[obj_name].joint_pos = np.array(i_j_pos)[None, ...].repeat(self.num_envs, axis=0)
                else:
                    template[obj_name].root_state = torch.tensor(root_state, device=self.device)[None, ...].repeat(
                        self.num_envs
                    )
                    template[obj_name].joint_pos = torch.tensor(i_j_pos, device=self.device)[None, ...].repeat(
                        self.num_envs
                    )
            self._cache["_initial_states"] = deepcopy(template)
        return cast(StatesType, self._cache["_initial_states"])

    @property
    def robot_names(self) -> list[str]:
        """Get the robot names in the environment."""
        if "_robot_names" not in self._cache:
            self._cache["_robot_names"] = list(self.robots.keys())
        return cast(list[str], self._cache["_robot_names"])

    @property
    def robots(self) -> dict[str, RobotModel]:
        """Get the robot models in the environment."""
        if "_robots" not in self._cache:
            self._cache["_robots"] = {
                name: RobotModel(cfg) for name, cfg in self.cfg_objects.items() if cfg.joints is not None
            }
        return cast(dict[str, RobotModel], self._cache["_robots"])

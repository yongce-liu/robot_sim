# """Sensor module for camera, IMU, and other sensors."""

# from abc import ABC, abstractmethod
# from collections import deque
# from dataclasses import MISSING
# from enum import Enum
# from typing import TYPE_CHECKING

# import numpy as np
# import torch
# from loguru import logger

# from robot_sim.utils import configclass
# from .simulator import BackendType

# if TYPE_CHECKING:
#     from robot_sim.backends import BaseBackend


# class SensorType(Enum):
#     """Enumeration of available sensor types."""

#     CAMERA = "camera"
#     CONTACT = "contact"


# # Registry to map sensor type to concrete class
# _SENSOR_TYPE_REGISTRY: dict[SensorType, type] = {}


# @configclass
# class SensorConfig(ABC):
#     """Base class for all sensors."""

#     type: SensorType = MISSING
#     """Type of the sensor."""
#     freq: float | None = None
#     """Update frequency in Hz. It should less than or equal to the simulation frequency."""
#     data_buffer_length: int = 1
#     """Maximum length of the data queue."""
#     ################### private attributes ###################
#     _data: torch.Tensor | np.ndarray | None = None
#     """the latest sensor data."""
#     _data_queue: deque[torch.Tensor | np.ndarray] | None = None
#     """Current sensor data."""
#     _last_update_cnt_stamp: int = 0
#     """Last update count stamp."""
#     _update_interval: int = 1
#     """Update interval in simulation steps."""
#     _backend: "BaseBackend | None" = None
#     """Backend simulator instance reference."""

#     @classmethod
#     def from_dict(cls, cfg_dict: dict) -> "SensorConfig":
#         """Override from_dict to handle polymorphic sensor types."""
#         if cls is not SensorConfig:
#             # If called on a concrete subclass, use default behavior
#             return super().from_dict(cfg_dict)

#         # If called on base class, determine concrete type from 'type' field
#         if "type" not in cfg_dict:
#             raise ValueError("Sensor configuration must include 'type' field")

#         sensor_type_str = cfg_dict["type"]
#         sensor_type = SensorType(sensor_type_str)

#         if sensor_type not in _SENSOR_TYPE_REGISTRY:
#             raise ValueError(
#                 f"Unknown sensor type: {sensor_type}. Available types: {list(_SENSOR_TYPE_REGISTRY.keys())}"
#             )

#         concrete_class = _SENSOR_TYPE_REGISTRY[sensor_type]
#         return concrete_class.from_dict(cfg_dict)

#     def _post_init__(self):
#         self._data_queue = deque(maxlen=self.data_buffer_length)

#     def _bind(self, *args, **kwargs) -> None:
#         raise NotImplementedError

#     def bind(self, backend: "BaseBackend", *args, **kwargs) -> None:
#         self._backend = backend
#         # Compute update interval based on frequency, if not specified, update every step
#         self._update_interval = int(backend._sim_freq / self.freq) if self.freq is not None else 1
#         assert self._update_interval > 0, "Sensor update frequency must be less than or equal to simulation frequency."
#         self._bind(*args, **kwargs)

#     def __call__(self, cnt: int, *args, **kwargs) -> torch.Tensor | np.ndarray | None:
#         """Update sensor data if frequency allows.

#         Args:
#             dt: Time delta since last call (in seconds)
#             *args, **kwds: Additional arguments for update method
#         """
#         # cnt: [0, self._backend._sim_freq-1]
#         # scenerio: cnt=10 last_cnt=490
#         if (cnt - self._last_update_cnt_stamp) % self._update_interval == 0:
#             self.update(*args, **kwargs)
#             self._data_queue.append(self._data)
#             self._last_update_cnt_stamp = cnt
#         return self.data

#     @abstractmethod
#     def update(self, *args, **kwargs) -> None:
#         """Update sensor data from the backend simulator.
#         Especially, you only need to update the _data attribute here for different simulator backends.
#         """
#         raise NotImplementedError

#     @property
#     def data(self) -> torch.Tensor | np.ndarray | None:
#         """Get the latest sensor data."""
#         return self._data

#     @property
#     def data_queue(self) -> deque[torch.Tensor | np.ndarray] | None:
#         """Get the data queue."""
#         return self._data_queue


# @configclass
# class Camera(SensorConfig):
#     """Camera sensor for RGB, depth, and segmentation."""

#     type: SensorType = SensorType.CAMERA
#     """Sensor type, defaults to CAMERA."""
#     width: int = 640
#     """Image width in pixels."""
#     height: int = 480
#     """Image height in pixels."""

#     mount_to: str | None = None
#     """Mount the camera to a specific object or robot. Defaults to None (world frame camera)."""

#     # when you want to mount the camera to a robot or object, set mount_to and mount_link
#     position: list[float] | None = None
#     """Camera position [x, y, z] in world frame. Used when camera is not mounted."""
#     look_at: list[float] | None = None
#     """Camera look-at point [x, y, z] in world frame. Used when camera is not mounted."""

#     # when you want to mount the camera to a robot or object, set mount_to and mount_link
#     mount_link: str | None = None
#     """Specify the link name to mount the camera to. Defaults to None."""
#     mount_pos: list[float] | None = None
#     """Position of the camera relative to the mount link. Defaults to (0, 0, 0)."""
#     mount_quat: list[float] | None = None
#     """Quaternion [w, x, y, z] of the camera relative to the mount link. Defaults to (1, 0, 0, 0)."""

#     # camera parameters
#     vertical_fov: float = 45.0
#     """Vertical field of view in degrees."""
#     data_types: list[str] = ["rgb"]
#     """Data types to capture: ['rgb', 'depth', 'segmentation']."""

#     def _post_init__(self):
#         super()._post_init__()

#         # Validate camera configuration
#         if self.mount_to is not None:
#             assert self.position is None, "position should not be set when mount_to is specified."
#             assert self.look_at is None, "look_at should not be set when mount_to is specified."
#             # Mounted camera: require mount_to and mount_link
#             assert self.mount_link is not None, "mount_link must be specified when mount_to is set."
#             if self.mount_pos is None:
#                 self.mount_pos = [0.0, 0.0, 0.0]
#             if self.mount_quat is None:
#                 self.mount_quat = [1.0, 0.0, 0.0, 0.0]  # [w, x, y, z]
#         else:
#             # World frame camera: require pos and look_at
#             assert self.position is not None, "position must be specified for world frame camera."
#             assert self.look_at is not None, "look_at must be specified for world frame camera."
#             logger.info(f"World frame camera at position {self.position} looking at {self.look_at}.")

#     def _bind(self, *args, **kwargs) -> None:
#         """Bind to mujoco backend and setup camera."""
#         if self.mount_link is not None:
#             self._setup_mounted_camera()
#             logger.info(
#                 f"Camera mounted to '{self.mount_to}' at link '{self.mount_link}', mount position {self.mount_pos} and orientation {self.mount_quat}."
#             )
#         else:
#             self._setup_world_camera()
#             logger.info(f"World frame camera at position {self.position} looking at {self.look_at}.")

#     def update(self, *args, **kwargs) -> None:
#         """Update camera data from mujoco backend."""
#         if self._backend is None:
#             raise RuntimeError("Backend not bound. Call bind() first.")

#         if self._backend.type == BackendType.MUJOCO:
#             self._update_mujoco()
#         else:
#             raise NotImplementedError(f"Camera update not implemented for backend: {self._backend.type}")

#     def _setup_world_camera(self, mjcf_model) -> None:
#         """Setup a world-frame camera using pos and look_at."""
#         # Compute camera orientation from pos and look_at
#         direction = np.array(
#             [
#                 self.look_at[0] - self.pos[0],
#                 self.look_at[1] - self.pos[1],
#                 self.look_at[2] - self.pos[2],
#             ]
#         )
#         direction = direction / np.linalg.norm(direction)
#         up = np.array([0, 0, 1])
#         right = np.cross(direction, up)
#         right = right / np.linalg.norm(right)
#         up = np.cross(right, direction)

#         camera_params = {
#             "pos": f"{self.pos[0]} {self.pos[1]} {self.pos[2]}",
#             "mode": "fixed",
#             "fovy": self.vertical_fov,
#             "xyaxes": f"{right[0]} {right[1]} {right[2]} {up[0]} {up[1]} {up[2]}",
#         }
#         mjcf_model.worldbody.add("camera", name=self._camera_id, **camera_params)

#     def _setup_mounted_camera(self, backend: "BaseBackend", mjcf_model) -> None:
#         """Setup a camera mounted to a specific link."""
#         # Find the target body (link) to mount the camera
#         model_name = backend._mjcf_sub_models.get(self.mount_to)
#         if model_name is None:
#             raise ValueError(f"Mount target '{self.mount_to}' not found in the model.")

#         # Find the specific link body
#         target_body = None
#         for body in model_name.find_all("body"):
#             if body.name == self.mount_link:
#                 target_body = body
#                 break

#         if target_body is None:
#             raise ValueError(f"Link '{self.mount_link}' not found in '{self.mount_to}'.")

#         # Convert quaternion [w, x, y, z] to rotation matrix, then to xyaxes
#         qw, qx, qy, qz = self.mount_quat

#         # Quaternion to rotation matrix
#         R = np.array(
#             [
#                 [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
#                 [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
#                 [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)],
#             ]
#         )

#         # Extract right and up vectors (MuJoCo camera convention)
#         right = R[:, 0]  # X-axis
#         up = R[:, 1]  # Y-axis

#         camera_params = {
#             "pos": f"{self.mount_pos[0]} {self.mount_pos[1]} {self.mount_pos[2]}",
#             "mode": "fixed",
#             "fovy": self.vertical_fov,
#             "xyaxes": f"{right[0]} {right[1]} {right[2]} {up[0]} {up[1]} {up[2]}",
#         }
#         target_body.add("camera", name=self._camera_id, **camera_params)

#     def _setup_mujoco_camera(self, backend: "BaseBackend") -> None:
#         """Setup camera in mujoco model."""
#         mjcf_model = backend._mjcf_model
#         if mjcf_model is None:
#             raise RuntimeError("MuJoCo model not initialized. Call backend._launch() first.")

#         if self.mount_to is not None:
#             # Mounted camera: attach to specified link
#             self._setup_mounted_camera(backend, mjcf_model)
#         else:
#             # World frame camera: use pos and look_at
#             self._setup_world_camera(mjcf_model)

#     def _update_mujoco(self) -> None:
#         """Capture camera data from mujoco."""
#         physics = self._backend._mjcf_physics

#         data_dict = {}

#         if "rgb" in self.data_types:
#             rgb = physics.render(width=self.width, height=self.height, camera_id=self._camera_id, depth=False)
#             data_dict["rgb"] = torch.from_numpy(rgb).float()

#         if "depth" in self.data_types:
#             depth = physics.render(width=self.width, height=self.height, camera_id=self._camera_id, depth=True)
#             data_dict["depth"] = torch.from_numpy(depth).float()

#         if "segmentation" in self.data_types:
#             seg = physics.render(
#                 width=self.width, height=self.height, camera_id=self._camera_id, depth=False, segmentation=True
#             )
#             # Extract geom IDs (first channel if multi-channel)
#             if seg.ndim == 3:
#                 seg = seg[..., 0]
#             data_dict["segmentation"] = torch.from_numpy(seg).long()

#         self._data = data_dict


# @configclass
# class ContactSensor(SensorConfig):
#     """Contact/force sensor."""

#     type: SensorType = SensorType.CONTACT
#     """Sensor type, defaults to CONTACT."""
#     history_length: int = 3
#     """Length of contact force history."""
#     _current_contact_force: torch.Tensor | None = None
#     """Current contact force."""
#     _contact_forces_queue: deque[torch.Tensor] | None = None
#     """Queue of contact forces."""

#     def _post_init__(self):
#         super()._post_init__()
#         self._contact_forces_queue = deque(maxlen=self.history_length)

#     # def bind_handler(self, handler: BaseSimHandler, *args, **kwargs):
#     #     """Bind the simulator handler and pre-compute per-robot indexing."""
#     #     super().bind_handler(handler, *args, **kwargs)
#     #     self.simulator = handler.scenario.simulator
#     #     self.num_envs = handler.scenario.num_envs
#     #     self.robots = handler.robots
#     #     if self.simulator in ["isaacgym", "mujoco"]:
#     #         self.body_ids_reindex = handler._get_body_ids_reindex(self.robots[0].name)
#     #     elif self.simulator == "isaacsim":
#     #         sorted_body_names = self.handler.get_body_names(self.robots[0].name, True)
#     #         self.body_ids_reindex = torch.tensor(
#     #             [self.handler.contact_sensor.body_names.index(name) for name in sorted_body_names],
#     #             dtype=torch.int,
#     #             device=self.handler.device,
#     #         )
#     #     else:
#     #         raise NotImplementedError
#     #     self.initialize()
#     #     self.__call__()

#     # def initialize(self):
#     #     """Warm-start the queue with `history_length` entries."""
#     #     for _ in range(self.history_length):
#     #         if self.simulator == "isaacgym":
#     #             self._current_contact_force = isaacgym.gymtorch.wrap_tensor(
#     #                 self.handler.gym.acquire_net_contact_force_tensor(self.handler.sim)
#     #             )
#     #         elif self.simulator == "isaacsim":
#     #             self._current_contact_force = self.handler.contact_sensor.data.net_forces_w
#     #         elif self.simulator == "mujoco":
#     #             self._current_contact_force = self._get_contact_forces_mujoco()
#     #         else:
#     #             raise NotImplementedError
#     #         self._contact_forces_queue.append(
#     #             self._current_contact_force.clone().view(self.num_envs, -1, 3)[:, self.body_ids_reindex, :]
#     #         )

#     # def _get_contact_forces_mujoco(self) -> torch.Tensor:
#     #     """Compute net contact forces on each body.

#     #     Returns:
#     #         torch.Tensor: shape (nbody, 3), contact forces for each body
#     #     """
#     #     nbody = self.handler.physics.model.nbody
#     #     contact_forces = torch.zeros((nbody, 3), device=self.handler.device)

#     #     for i in range(self.handler.physics.data.ncon):
#     #         contact = self.handler.physics.data.contact[i]
#     #         force = np.zeros(6, dtype=np.float64)
#     #         mujoco.mj_contactForce(self.handler.physics.model.ptr, self.handler.physics.data.ptr, i, force)
#     #         f_contact = torch.from_numpy(force[:3]).to(device=self.handler.device)

#     #         body1 = self.handler.physics.model.geom_bodyid[contact.geom1]
#     #         body2 = self.handler.physics.model.geom_bodyid[contact.geom2]

#     #         contact_forces[body1] += f_contact
#     #         contact_forces[body2] -= f_contact

#     #     return contact_forces

#     # def __call__(self):
#     #     """Fetch the newest net contact forces and update the queue."""
#     #     if self.simulator == "isaacgym":
#     #         self.handler.gym.refresh_net_contact_force_tensor(self.handler.sim)
#     #     elif self.simulator == "isaacsim":
#     #         self._current_contact_force = self.handler.contact_sensor.data.net_forces_w
#     #     elif self.simulator == "mujoco":
#     #         self._current_contact_force = self._get_contact_forces_mujoco()
#     #     else:
#     #         raise NotImplementedError
#     #     self._contact_forces_queue.append(
#     #         self._current_contact_force.view(self.num_envs, -1, 3)[:, self.body_ids_reindex, :]
#     #     )
#     #     return {self.robots[0].name: self}

#     # @property
#     # def contact_forces_history(self) -> torch.Tensor:
#     #     """Return stacked history as (num_envs, history_length, num_bodies, 3)."""
#     #     return torch.stack(list(self._contact_forces_queue), dim=1)  # (num_envs, history_length, num_bodies, 3)

#     # @property
#     # def contact_forces(self) -> torch.Tensor:
#     #     """Return the latest contact forces snapshot."""
#     #     return self._contact_forces_queue[-1]


# # Register sensor types
# _SENSOR_TYPE_REGISTRY[SensorType.CAMERA] = Camera
# _SENSOR_TYPE_REGISTRY[SensorType.CONTACT] = ContactSensor

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from robot_sim.configs import BackendType

from .base import BaseSensor

if TYPE_CHECKING:
    from robot_sim.backends import MujocoBackend


class Camera(BaseSensor):
    """Camera sensor for RGB, depth, and segmentation."""

    _backend: "MujocoBackend | None" = None
    """Backend simulator instance reference."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self._data = defaultdict(lambda: None)
        """Camera data dictionary with keys: 'rgb', 'depth', 'segmentation'."""

    def _bind(self, obj_name: str, sensor_name: str, **kwargs) -> None:
        """Bind to mujoco backend and setup camera."""
        self.obj_name = obj_name
        self.sensor_name = sensor_name
        if self.config.mount_to is not None:
            if self._backend.type == BackendType.MUJOCO:
                self._setup_mounted_camera_mujoco()
            else:
                logger.error(f"Mounted camera not supported for backend: {self._backend.type}")
        else:
            if self._backend.type == BackendType.MUJOCO:
                self._setup_world_camera_mujoco()
            else:
                logger.error(f"World camera not supported for backend: {self._backend.type}")
        if self._backend.type == BackendType.MUJOCO:
            self._camera_id = f"{self.obj_name}/{self.sensor_name}"
        logger.info(
            f"Initializing Camera Sensor: width={self.config.width}, height={self.config.height}. Camera {sensor_name} will be mounted to '{self.config.mount_to}' at link '{self.config.mount_to}' for object {obj_name} with position {self.config.position} and quaternion {self.config.orientation}."
        )

    def _update(self) -> None:
        """Update camera data from mujoco backend."""
        if self._backend.type == BackendType.MUJOCO:
            self._update_mujoco()
        else:
            raise NotImplementedError(f"Camera update not implemented for backend: {self._backend.type}")

    def _setup_world_camera_mujoco(self) -> None:
        """Setup a world-frame camera using pos and look_at."""
        # Compute camera orientation from pos and look_at
        mjcf_model = self._backend._mjcf_model
        direction = np.array(
            [
                self.look_at[0] - self.pos[0],
                self.look_at[1] - self.pos[1],
                self.look_at[2] - self.pos[2],
            ]
        )
        direction = direction / np.linalg.norm(direction)
        up = np.array([0, 0, 1])
        right = np.cross(direction, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, direction)

        camera_params = {
            "pos": f"{self.pos[0]} {self.pos[1]} {self.pos[2]}",
            "mode": "fixed",
            "fovy": self.vertical_fov,
            "xyaxes": f"{right[0]} {right[1]} {right[2]} {up[0]} {up[1]} {up[2]}",
        }
        mjcf_model.worldbody.add("camera", name=self.sensor_name, **camera_params)

    def _setup_mounted_camera_mujoco(self) -> None:
        """Setup a camera mounted to a specific link."""
        # Find the target body (link) to mount the camera
        model = self._backend._mjcf_model
        if model is None:
            raise ValueError(f"Mount target '{self.config.mount_to}' not found in the model.")

        # Find the specific link body
        target_body = None
        for body_name in self._backend.get_body_names(self.obj_name):
            if body_name == f"{self.obj_name}/{self.config.mount_to}":
                target_body = model.find("body", body_name)
                break

        if target_body is None:
            raise ValueError(f"Link '{self.config.mount_to}' not found in '{self.obj_name}'.")

        camera_params = {
            "pos": f"{self.config.position[0]} {self.config.position[1]} {self.config.position[2]}",
            "mode": "fixed",
            "fovy": self.config.vertical_fov,
            "quat": f"{self.config.orientation[0]} {self.config.orientation[1]} {self.config.orientation[2]} {self.config.orientation[3]}",
            # "euler": "0 -0.8 -1.57",  # in radians
        }
        # logger.info(f"euler angles (rad): roll={roll.item()}, pitch={pitch.item()}, yaw={yaw.item()}")
        target_body.add("camera", name=self.sensor_name, **camera_params)

    def _update_mujoco(self) -> None:
        """Capture camera data from mujoco."""
        physics = self._backend._mjcf_physics

        if "rgb" in self.config.data_types:
            self._data["rgb"] = physics.render(
                width=self.config.width, height=self.config.height, camera_id=self._camera_id, depth=False
            )

        if "depth" in self.config.data_types:
            self._data["depth"] = physics.render(
                width=self.config.width, height=self.config.height, camera_id=self._camera_id, depth=True
            )

        if "segmentation" in self.config.data_types:
            seg = physics.render(
                width=self.config.width,
                height=self.config.height,
                camera_id=self._camera_id,
                depth=False,
                segmentation=True,
            )
            # Extract geom IDs (first channel if multi-channel)
            if seg.ndim == 3:
                seg = seg[..., 0]
            self._data["segmentation"] = seg

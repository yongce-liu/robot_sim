from typing import TYPE_CHECKING

import numpy as np
import torch
from loguru import logger

from robot_sim.configs import BackendType, CameraConfig
from robot_sim.utils.math import euler_xyz_from_quat

from .base import BaseSensor

if TYPE_CHECKING:
    from robot_sim.backends import MujocoBackend


class Camera(BaseSensor):
    """Camera sensor for RGB, depth, and segmentation."""

    _backend: "MujocoBackend | None" = None
    """Backend simulator instance reference."""

    def __init__(self, config: CameraConfig, **kwargs):
        super().__init__(config, **kwargs)
        logger.info(
            f"Initializing Camera Sensor: width={self.config.width}, height={self.config.height}"
            f"Camera will be mounted to '{self.config.mount_to}' at link '{self.config.mount_to}' "
            f"with position {self.config.position} and quaternion {self.config.orientation}."
        )

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

    def update(self) -> None:
        """Update camera data from mujoco backend."""
        if self._backend is None:
            raise RuntimeError("Backend not bound. Call bind() first.")

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
        mjcf_model.worldbody.add("camera", name=self._camera_id, **camera_params)

    def _setup_mounted_camera_mujoco(self) -> None:
        """Setup a camera mounted to a specific link."""
        # Find the target body (link) to mount the camera
        model_name = self._backend._mjcf_sub_models.get(self.obj_name)
        if model_name is None:
            raise ValueError(f"Mount target '{self.config.mount_to}' not found in the model.")

        # Find the specific link body
        target_body = None
        for body in model_name.find_all("body"):
            if body.name == self.config.mount_to:
                target_body = body
                break

        if target_body is None:
            raise ValueError(f"Link '{self.config.mount_to}' not found in '{self.obj_name}'.")

        # Convert quaternion [w, x, y, z] to rotation matrix, then to xyaxes
        roll, pitch, yaw = euler_xyz_from_quat(
            torch.tensor([self.config.orientation], dtype=torch.float32)
        )  # Shape (1, 3)
        camera_params = {
            "pos": f"{self.config.position[0]} {self.config.position[1]} {self.config.position[2]}",
            "mode": "fixed",
            "fovy": self.config.vertical_fov,
            "euler": f"{roll.item()} {pitch.item()} {yaw.item()}",  # in radians
        }
        # logger.info(f"euler angles (rad): roll={roll.item()}, pitch={pitch.item()}, yaw={yaw.item()}")
        target_body.add("camera", name=self.sensor_name, **camera_params)

    def _setup_mujoco_camera(self) -> None:
        """Setup camera in mujoco model."""
        mjcf_model = self._backend._mjcf_model
        if mjcf_model is None:
            raise RuntimeError("MuJoCo model not initialized. Call backend._launch() first.")

        if self.mount_to is not None:
            # Mounted camera: attach to specified link
            self._setup_mounted_camera()
        else:
            # World frame camera: use pos and look_at
            self._setup_world_camera(mjcf_model)

    def _update_mujoco(self) -> None:
        """Capture camera data from mujoco."""
        physics = self._backend._mjcf_physics

        data_dict = {}

        if "rgb" in self.data_types:
            rgb = physics.render(width=self.width, height=self.height, camera_id=self._camera_id, depth=False)
            data_dict["rgb"] = torch.from_numpy(rgb).float()

        if "depth" in self.data_types:
            depth = physics.render(width=self.width, height=self.height, camera_id=self._camera_id, depth=True)
            data_dict["depth"] = torch.from_numpy(depth).float()

        if "segmentation" in self.data_types:
            seg = physics.render(
                width=self.width, height=self.height, camera_id=self._camera_id, depth=False, segmentation=True
            )
            # Extract geom IDs (first channel if multi-channel)
            if seg.ndim == 3:
                seg = seg[..., 0]
            data_dict["segmentation"] = torch.from_numpy(seg).long()

        self._data = data_dict

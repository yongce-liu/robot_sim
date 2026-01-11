from collections import defaultdict

import numpy as np
from loguru import logger

from robot_sim.configs import BackendType, CameraConfig

from .base import BaseSensor


class Camera(BaseSensor):
    """Camera sensor for RGB, depth, and segmentation."""

    config: CameraConfig
    """Camera sensor configuration for the camera."""

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
            f"Initializing Camera Sensor: width={self.config.width}, height={self.config.height}. Camera {sensor_name} will be mounted to '{self.config.mount_to}' at link '{self.config.mount_to}' for object {obj_name} with position {self.config.pose[:3]} and quaternion {self.config.pose[3:]}."
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
        lookat, pose = self.config.look_at, self.config.pose
        assert lookat is not None, "look_at must be specified for world camera."
        direction = np.array([lookat[0] - pose[0], lookat[1] - pose[1], lookat[2] - pose[2]])
        direction = direction / np.linalg.norm(direction)
        up = np.array([0, 0, 1])
        right = np.cross(direction, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, direction)

        camera_params = {
            "pos": f"{pose[0]} {pose[1]} {pose[2]}",
            "mode": "fixed",
            "fovy": self.config.vertical_fov,
            "xyaxes": f"{right[0]} {right[1]} {right[2]} {up[0]} {up[1]} {up[2]}",
        }
        mjcf_model.worldbody.add("camera", name=self.sensor_name, **camera_params)

    def _setup_mounted_camera_mujoco(self) -> None:
        """Setup a camera mounted to a specific link."""
        # Find the target body (link) to mount the camera
        model = self._backend._mjcf_model
        assert model is not None, "Mujoco model is not initialized in the backend."
        pose = self.config.pose
        # Find the specific link body
        target_body = None
        for body_name in self._backend.objects[self.obj_name].get_body_names(prefix=self.obj_name + "/"):
            if body_name == f"{self.obj_name}/{self.config.mount_to}":
                target_body = model.find("body", body_name)
                break

        if target_body is None:
            raise ValueError(f"Link '{self.config.mount_to}' not found in '{self.obj_name}'.")

        camera_params = {
            "pos": f"{pose[0]} {pose[1]} {pose[2]}",
            "mode": "fixed",
            "fovy": self.config.vertical_fov,
            "quat": f"{pose[3]} {pose[4]} {pose[5]} {pose[6]}",
            # "euler": "0 -0.8 -1.57",  # in radians
        }
        # logger.info(f"euler angles (rad): roll={roll.item()}, pitch={pitch.item()}, yaw={yaw.item()}")
        target_body.add("camera", name=self.sensor_name, **camera_params)

    def _update_mujoco(self, dtype=np.uint8) -> None:
        """Capture camera data from mujoco."""
        physics = self._backend._mjcf_physics

        if "rgb" in self.config.data_types:
            self._data["rgb"] = physics.render(
                width=self.config.width, height=self.config.height, camera_id=self._camera_id, depth=False
            ).astype(dtype)

        if "depth" in self.config.data_types:
            self._data["depth"] = physics.render(
                width=self.config.width, height=self.config.height, camera_id=self._camera_id, depth=True
            ).astype(dtype)

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
            self._data["segmentation"] = seg.astype(np.int32)

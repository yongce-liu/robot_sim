import numpy as np
import torch
from loguru import logger

from robot_sim.configs import CameraConfig

from .base import BaseSensor


class Camera(BaseSensor):
    """Camera sensor for RGB, depth, and segmentation."""

    def __init__(self, config: CameraConfig, **kwargs):
        super().__init__(config, **kwargs)

    def _bind(self, *args, **kwargs) -> None:
        pass
        # """Bind to mujoco backend and setup camera."""
        # if self.mount_link is not None:
        #     self._setup_mounted_camera()
        #     logger.info(
        #         f"Camera mounted to '{self.mount_to}' at link '{self.mount_link}', mount position {self.mount_pos} and orientation {self.mount_quat}."
        #     )
        # else:
        #     self._setup_world_camera()
        #     logger.info(f"World frame camera at position {self.position} looking at {self.look_at}.")

    def update(self, *args, **kwargs) -> None:
        """Update camera data from mujoco backend."""
        if self._backend is None:
            raise RuntimeError("Backend not bound. Call bind() first.")

        if self._backend.type == BackendType.MUJOCO:
            self._update_mujoco()
        else:
            raise NotImplementedError(f"Camera update not implemented for backend: {self._backend.type}")

    def _setup_world_camera(self, mjcf_model) -> None:
        """Setup a world-frame camera using pos and look_at."""
        # Compute camera orientation from pos and look_at
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

    def _setup_mounted_camera(self) -> None:
        """Setup a camera mounted to a specific link."""
        # Find the target body (link) to mount the camera
        model_name = self._backend._mjcf_sub_models.get(self.mount_to)
        if model_name is None:
            raise ValueError(f"Mount target '{self.mount_to}' not found in the model.")

        # Find the specific link body
        target_body = None
        for body in model_name.find_all("body"):
            if body.name == self.mount_link:
                target_body = body
                break

        if target_body is None:
            raise ValueError(f"Link '{self.mount_link}' not found in '{self.mount_to}'.")

        # Convert quaternion [w, x, y, z] to rotation matrix, then to xyaxes
        qw, qx, qy, qz = self.mount_quat

        # Quaternion to rotation matrix
        R = np.array(
            [
                [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
                [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
                [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)],
            ]
        )

        # Extract right and up vectors (MuJoCo camera convention)
        right = R[:, 0]  # X-axis
        up = R[:, 1]  # Y-axis

        camera_params = {
            "pos": f"{self.mount_pos[0]} {self.mount_pos[1]} {self.mount_pos[2]}",
            "mode": "fixed",
            "fovy": self.vertical_fov,
            "xyaxes": f"{right[0]} {right[1]} {right[2]} {up[0]} {up[1]} {up[2]}",
        }
        target_body.add("camera", name=self._camera_id, **camera_params)

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

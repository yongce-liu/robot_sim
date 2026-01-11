import time
from collections import defaultdict

import numpy as np
import torch
from loguru import logger

from robot_sim.configs import BackendType, CameraConfig

from .base import BaseSensor


class Camera(BaseSensor):
    """Camera sensor for RGB, depth, and segmentation."""

    config: CameraConfig
    """Camera sensor configuration for the camera."""
    _data: dict[str, torch.Tensor | np.ndarray]
    """Camera data dictionary with keys: 'rgb', 'depth', 'segmentation'."""

    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        self._data = defaultdict(lambda: None)
        """Camera data dictionary with keys: 'rgb', 'depth', 'segmentation'."""

    def _bind(self, obj_name: str, sensor_name: str, **kwargs) -> None:
        """Bind to mujoco backend and setup camera."""
        self.obj_name = obj_name
        self.sensor_name = sensor_name
        if self._backend.type == BackendType.MUJOCO:
            self._setup_mujoco()
        else:
            self._setup_hardware()

    def _update(self) -> None:
        """Update camera data from mujoco backend."""
        if self._backend.type == BackendType.MUJOCO:
            self._update_mujoco()
        else:
            self._update_hardware()

    ################### MUJOCO BACKEND ###################
    def _setup_mujoco(self) -> None:
        if hasattr(self._backend, "_mjcf_model"):
            mjcf_model = self._backend._mjcf_model
        else:
            raise ValueError("Mujoco model is not initialized in the backend.")
        if self.config.mount_to is not None:
            self._setup_mounted_camera_mujoco(model=mjcf_model)
        else:
            self._setup_world_camera_mujoco(model=mjcf_model)
        self._camera_id = f"{self.obj_name}/{self.sensor_name}"

        logger.info(
            f"Initializing Camera Sensor: width={self.config.width}, height={self.config.height}. Camera {self.sensor_name} will be mounted to '{self.config.mount_to}' at link '{self.config.mount_to}' for object {self.obj_name} with position {self.config.pose[:3]} and quaternion {self.config.pose[3:]}."
        )

    def _setup_world_camera_mujoco(self, model) -> None:
        """Setup a world-frame camera using pos and look_at."""
        # Compute camera orientation from pos and look_at

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
        model.worldbody.add("camera", name=self.sensor_name, **camera_params)

    def _setup_mounted_camera_mujoco(self, model) -> None:
        """Setup a camera mounted to a specific link."""
        # Find the target body (link) to mount the camera
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
        physics = getattr(self._backend, "_mjcf_physics")

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

    ################### HARDWARE EXP ###################
    def _setup_hardware(self) -> None:
        if self.config.series is None:
            logger.warning(
                f"Camera {self.sensor_name} has no series configured; skipping hardware binding and this camera is not used."
            )
            return

        series = self.config.series.lower()
        if series == "realsense":
            self._setup_realsense()
        else:
            logger.warning(
                f"Camera {self.sensor_name} series '{self.config.series}' is not supported; this camera is not used."
            )
            return

        logger.info(
            f"Initializing Hardware Camera Sensor: series={self.config.series}, width={self.config.width}, height={self.config.height}."
        )

    def _setup_realsense(self) -> None:
        """Setup Intel RealSense streams for hardware experiments."""

        import pyrealsense2 as rs

        self._rs_align: rs.align | None = None
        self._rs_pipeline: rs.pipeline | None = None

        fps = int(self.config.freq) if self.config.freq is not None else 30
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.config.width, self.config.height, rs.format.bgr8, fps)
        if "depth" in self.config.data_types:
            config.enable_stream(rs.stream.depth, self.config.width, self.config.height, rs.format.z16, fps)

        profile = pipeline.start(config)
        if "depth" in self.config.data_types:
            depth_sensor = profile.get_device().first_depth_sensor()
            self._rs_depth_scale = depth_sensor.get_depth_scale()
            self._rs_align = rs.align(rs.stream.color)

        self._rs_pipeline = pipeline
        time.sleep(2.0)  # Allow camera to warm up

    def _update_hardware(self) -> None:
        """Capture camera data from hardware devices."""
        if self.config.series is None:
            return

        series = self.config.series.lower()
        if series == "realsense" and self._rs_pipeline is not None:
            frames = self._rs_pipeline.wait_for_frames()
            if self._rs_align is not None:
                frames = self._rs_align.process(frames)

            if "rgb" in self.config.data_types:
                color_frame = frames.get_color_frame()
                if color_frame:
                    self._data["rgb"] = np.asanyarray(color_frame.get_data())

            if "depth" in self.config.data_types:
                depth_frame = frames.get_depth_frame()
                if depth_frame:
                    depth = np.asanyarray(depth_frame.get_data())
                    if self._rs_depth_scale is not None:
                        depth = depth.astype(np.float32) * self._rs_depth_scale
                    self._data["depth"] = depth
        else:
            logger.warning(f"Camera {self.sensor_name} series '{self.config.series}' is not supported for updates.")

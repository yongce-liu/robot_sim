import time
from collections import defaultdict

import cv2
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
        self.series = self.config.series.lower() if self.config.series is not None else None

        if self._backend.type == BackendType.MUJOCO:
            self._setup_mujoco()
        elif self.series == "realsense":
            self._setup_realsense()
        elif self.series == "zmq":
            self._setup_zmq()
        elif self.series == "ros2":
            self._setup_ros2()
        else:
            logger.error(f"Camera {self.sensor_name} series '{self.series}' is not supported; this camera is not used.")
            return
        logger.info(
            f"Initializing Camera Sensor: width={self.config.width}, height={self.config.height}. Camera name {self.sensor_name}, Series: {self.series} will be mounted to '{self.config.mount_to}' at link '{self.config.mount_to}' for object {self.obj_name} with position {self.config.pose[:3]} and quaternion {self.config.pose[3:]}."
        )

    def _update(self) -> None:
        """Update camera data from mujoco backend."""
        if self._backend.type == BackendType.MUJOCO:
            self._update_mujoco()
        elif self.series == "realsense":
            self._update_realsense()
        elif self.series == "zmq":
            self._update_zmq()
        elif self.series == "ros2":
            self._update_ros2()
        else:
            return None

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

    ################### PYREALSENSE2 ###################
    def _setup_realsense(self) -> None:
        """Setup Intel RealSense streams for hardware experiments."""
        import pyrealsense2 as rs

        fps = int(self.config.freq) if self.config.freq is not None else 30
        serial_number = self.config.extras.get("serial_number") if self.config.extras else None

        self._rs_align: rs.align | None = None
        self._rs_pipeline: rs.pipeline = rs.pipeline()

        config = rs.config()
        if serial_number is not None:
            config.enable_device(serial_number)
            logger.info(f"Using RealSense camera with serial number: {serial_number}")
        config.enable_stream(rs.stream.color, self.config.width, self.config.height, rs.format.bgr8, fps)
        if "depth" in self.config.data_types:
            config.enable_stream(rs.stream.depth, self.config.width, self.config.height, rs.format.z16, fps)

        profile = self._rs_pipeline.start(config)
        if "depth" in self.config.data_types:
            depth_sensor = profile.get_device().first_depth_sensor()
            self._rs_depth_scale = depth_sensor.get_depth_scale()
            self._rs_align = rs.align(rs.stream.color)

        time.sleep(2.0)  # Allow camera to warm up

    def _update_realsense(self) -> None:
        frames = self._rs_pipeline.wait_for_frames()
        if self._rs_align is not None:
            frames = self._rs_align.process(frames)

        if "rgb" in self.config.data_types:
            color_frame = frames.get_color_frame()
            if color_frame:
                self._data["rgb"] = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB)

        if "depth" in self.config.data_types:
            depth_frame = frames.get_depth_frame()
            if depth_frame:
                depth = np.asanyarray(depth_frame.get_data())
                if self._rs_depth_scale is not None:
                    depth = depth.astype(np.float32) * self._rs_depth_scale
                self._data["depth"] = depth

    ################### PYZMQ ###################
    def _setup_zmq(self) -> None:
        import zmq

        extras = self.config.extras or {}
        client_ip = extras.get("ip", "127.0.0.1")
        client_port = extras.get("port", 5555)
        endpoint = extras.get("endpoint", f"tcp://{client_ip}:{client_port}")
        self._zmq_unit_test = bool(extras.get("unit_test", False))

        context = zmq.Context.instance()
        socket = context.socket(zmq.SUB)
        socket.setsockopt(zmq.SUBSCRIBE, b"")
        socket.setsockopt(zmq.RCVHWM, 100)
        socket.connect(endpoint)

        self._zmq_sub = socket
        self._zmq_NOBLOCK = zmq.NOBLOCK
        self._zmq_AGAIN = zmq.Again

        now = time.time()
        timeout = extras.get("timeout", 5.0)
        while not self._update_zmq():
            if time.time() - now > timeout:
                raise TimeoutError(f"Could not connect to ZMQ camera at {endpoint} within {timeout} seconds.")
            time.sleep(0.1)
        time.sleep(2.0)  # Allow camera to warm up

    def _update_zmq(self) -> bool:
        try:
            parts = self._zmq_sub.recv_multipart(flags=self._zmq_NOBLOCK)  # rgb, depth, ...
        except self._zmq_AGAIN:
            return False
        if getattr(self, "_zmq_unit_test", False):
            parts = [part[12:] for part in parts]  # Remove unit test header
        if not parts:
            return False
        if "rgb" in self.config.data_types:
            self._data["rgb"] = cv2.cvtColor(
                cv2.imdecode(np.frombuffer(parts[0], dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
            )
        if "depth" in self.config.data_types and len(parts) > 1:
            self._data["depth"] = cv2.imdecode(np.frombuffer(parts[1], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        return True

    ################### ROS2 ###################
    def _setup_ros2(self) -> None:
        extras = self.config.extras or {}
        try:
            import rclpy
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import Image
        except ImportError as exc:
            raise ImportError("ROS2 dependencies (rclpy, sensor_msgs) are required for ros2 cameras.") from exc

        if not rclpy.ok():
            rclpy.init(args=None)

        node_name = extras.get(
            "node_name",
            f"robot_sim_camera_{self.obj_name}_{self.sensor_name}".replace("/", "_"),
        )
        self._ros2_node = rclpy.create_node(node_name)
        self._ros2_rclpy = rclpy
        self._ros2_image_dtype_map = {
            "rgb8": (np.uint8, 3),
            "bgr8": (np.uint8, 3),
            "rgba8": (np.uint8, 4),
            "bgra8": (np.uint8, 4),
            "mono8": (np.uint8, 1),
            "mono16": (np.uint16, 1),
            "16uc1": (np.uint16, 1),
            "32fc1": (np.float32, 1),
        }

        def _make_callback(data_key: str):
            def _callback(msg: Image) -> None:
                encoding = msg.encoding.lower()
                dtype_channels = self._ros2_image_dtype_map.get(encoding)
                if dtype_channels is None:
                    logger.warning(f"Unsupported ROS2 image encoding '{msg.encoding}' for camera {self.sensor_name}.")
                    return
                dtype, channels = dtype_channels
                itemsize = np.dtype(dtype).itemsize
                if channels == 1:
                    stride = msg.step // itemsize
                    frame = np.frombuffer(msg.data, dtype=dtype).reshape((msg.height, stride))[:, : msg.width]
                else:
                    stride = msg.step // (itemsize * channels)
                    frame = np.frombuffer(msg.data, dtype=dtype).reshape((msg.height, stride, channels))[
                        :, : msg.width, :
                    ]
                self._data[data_key] = frame

            return _callback

        self._ros2_subs = {}
        if "rgb" in self.config.data_types:
            rgb_topic = extras.get("rgb_topic")
            if rgb_topic:
                self._ros2_subs["rgb"] = self._ros2_node.create_subscription(
                    Image, rgb_topic, _make_callback("rgb"), qos_profile_sensor_data
                )
            else:
                logger.warning(f"ROS2 camera {self.sensor_name} missing extras['rgb_topic']; skipping rgb.")

        if "depth" in self.config.data_types:
            depth_topic = extras.get("depth_topic")
            if depth_topic:
                self._ros2_subs["depth"] = self._ros2_node.create_subscription(
                    Image, depth_topic, _make_callback("depth"), qos_profile_sensor_data
                )
            else:
                logger.warning(f"ROS2 camera {self.sensor_name} missing extras['depth_topic']; skipping depth.")

        if "segmentation" in self.config.data_types:
            seg_topic = extras.get("segmentation_topic")
            if seg_topic:
                self._ros2_subs["segmentation"] = self._ros2_node.create_subscription(
                    Image, seg_topic, _make_callback("segmentation"), qos_profile_sensor_data
                )
            else:
                logger.warning(
                    f"ROS2 camera {self.sensor_name} missing extras['segmentation_topic']; skipping segmentation."
                )

    def _update_ros2(self) -> None:
        self._ros2_rclpy.spin_once(self._ros2_node, timeout_sec=0.0)

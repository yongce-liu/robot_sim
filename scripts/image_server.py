import struct
import time
from collections import defaultdict, deque
from dataclasses import dataclass

import cv2
import numpy as np
import zmq
from loguru import logger


@dataclass
class CameraConfig:
    idx: int | None = None
    """Camera index or serial number"""
    type: str = "opencv"
    width: int = 640
    height: int = 480
    fps: int = 30
    enable_depth: bool = False
    port: int = 9999


class RealSenseCamera(object):
    def __init__(self, img_shape, fps, serial_number: int | None = None, enable_depth=False) -> None:
        """
        img_shape: [height, width]
        serial_number: serial number
        """
        self.img_shape = img_shape  # height, width
        self.fps = fps
        self.serial_number = str(serial_number) if serial_number is not None else None
        self.enable_depth = enable_depth

        self.init_realsense()

    def init_realsense(self):
        import pyrealsense2 as rs

        self.align = rs.align(rs.stream.color)
        self.pipeline = rs.pipeline()
        config = rs.config()
        if self.serial_number is not None:
            config.enable_device(self.serial_number)

        config.enable_stream(rs.stream.color, self.img_shape[1], self.img_shape[0], rs.format.bgr8, self.fps)

        if self.enable_depth:
            config.enable_stream(rs.stream.depth, self.img_shape[1], self.img_shape[0], rs.format.z16, self.fps)

        profile = self.pipeline.start(config)
        self._device = profile.get_device()
        if self._device is None:
            logger.error("[Image Server] pipe_profile.get_device() is None .")
        if self.enable_depth:
            assert self._device is not None
            depth_sensor = self._device.first_depth_sensor()
            self.g_depth_scale = depth_sensor.get_depth_scale()

        self.intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

    def get_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()

        if self.enable_depth:
            depth_frame = aligned_frames.get_depth_frame()

        if not color_frame:
            return None

        color_image = np.asanyarray(color_frame.get_data())
        # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        depth_image = np.asanyarray(depth_frame.get_data()) if self.enable_depth else None
        return color_image, depth_image

    def release(self):
        self.pipeline.stop()


class OpenCVCamera:
    def __init__(self, device_id: int | None, img_shape, fps):
        """
        decive_id: /dev/video* or *
        img_shape: [height, width]
        """
        if device_id is None:
            device_id = 0
            logger.warning("[Image Server] OpenCV camera device_id is None, using default device_id 0.")
        self.id = device_id
        self.fps = fps
        self.img_shape = img_shape
        self.cap = cv2.VideoCapture(self.id, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc("M", "J", "P", "G"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.img_shape[1])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.img_shape[0])
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Test if the camera can read frames
        now = time.time()
        timeout = 5.0
        while True:
            if time.time() - now > timeout:
                raise TimeoutError(f"Could not initialize OpenCV camera {self.id} within {timeout} seconds.")
            success, frame = self.cap.read()
            if self.check_frame(frame=frame) and success:
                break
            time.sleep(0.1)

    def check_frame(self, frame: np.ndarray) -> bool:
        if frame.ndim != 3 or frame.shape[2] != 3:
            self.release()
            logger.error(f"[Image Server] Camera {self.id} is not a 3-channel color stream, got shape={frame.shape}")
            return False
        h, w = frame.shape[:2]
        exp_h, exp_w = self.img_shape[0], self.img_shape[1]

        if (h, w) != (exp_h, exp_w):
            self.release()
            logger.error(
                f"Camera {self.id} size mismatch: got {h}x{w}, expected {exp_h}x{exp_w}. "
                f"(Driver may ignore cap.set; try another /dev/videoX or use realsense backend.)"
            )
            return False
        return True

    def release(self):
        self.cap.release()

    def get_frame(self):
        ret, color_image = self.cap.read()
        if not ret:
            return None
        return color_image, None


class ImageServer:
    def __init__(self, config: dict[str, CameraConfig], ip="0.0.0.0", Unit_Test=False):
        """Image server using ZeroMQ to publish images from multiple cameras."""
        self.config = config
        self.ip = ip
        self.Unit_Test = Unit_Test

        # Initialize head cameras
        self.cameras: dict[str, OpenCVCamera | RealSenseCamera] = {}
        self.camera_names: list[str] = []  # Initialize camera names list
        for cam_name, cam_cfg in config.items():
            logger.info(
                f"[Image Server] Camera {cam_name} ID {cam_cfg.idx} resolution: {cam_cfg.height} x {cam_cfg.width} at {cam_cfg.fps} FPS."
            )
            if cam_cfg.type == "opencv":
                if cam_cfg.enable_depth:
                    logger.warning(
                        f"[Image Server] OpenCV camera does not support depth. Camera {cam_name} depth will be disabled."
                    )
                self.cameras[cam_name] = OpenCVCamera(
                    device_id=cam_cfg.idx, img_shape=[cam_cfg.height, cam_cfg.width], fps=cam_cfg.fps
                )
            elif cam_cfg.type == "realsense":
                self.cameras[cam_name] = RealSenseCamera(
                    img_shape=[cam_cfg.height, cam_cfg.width],
                    fps=cam_cfg.fps,
                    serial_number=cam_cfg.idx,
                    enable_depth=cam_cfg.enable_depth,
                )
            else:
                logger.error(
                    f"[Image Server] Unsupported camera type: {cam_cfg.type} for camera {cam_name}. Supported types are 'opencv' and 'realsense'."
                )
        self.camera_names = list(self.cameras.keys())

        # Set ZeroMQ context and socket
        self.sockets: dict[str, zmq.Socket] = {}
        self.context = zmq.Context()
        used_ports = set()
        for cam_name, cam_cfg in config.items():
            port = cam_cfg.port
            while port in used_ports:
                logger.warning(
                    f"[Image Server] Port {port} for camera {cam_name} is already used by another camera. "
                    f"Automatically incrementing port number."
                )
                port += 1
            used_ports.add(port)
            self.sockets[cam_name] = self.context.socket(zmq.PUB)
            self.sockets[cam_name].bind(f"tcp://{self.ip}:{port}")
            logger.info(f"[Image Server] Camera {cam_name} is publishing on tcp://{self.ip}:{port}")

        if self.Unit_Test:
            self._init_performance_metrics()

        logger.info("[Image Server] Image server has started, waiting for client connections...")

    def _init_performance_metrics(self):
        self.frame_count = 0  # Total frames sent
        self.time_window = 1.0  # Time window for FPS calculation (in seconds)
        self.frame_times = deque()  # Timestamps of frames sent within the time window
        self.start_time = time.time()  # Start time of the streaming

    def _update_performance_metrics(self, current_time):
        # Add current time to frame times deque
        self.frame_times.append(current_time)
        # Remove timestamps outside the time window
        while self.frame_times and self.frame_times[0] < current_time - self.time_window:
            self.frame_times.popleft()
        # Increment frame count
        self.frame_count += 1

    def _print_performance_metrics(self, current_time):
        if self.frame_count % 30 == 0:
            elapsed_time = current_time - self.start_time
            real_time_fps = len(self.frame_times) / self.time_window
            logger.info(
                f"[Image Server] Real-time FPS: {real_time_fps:.2f}, Total frames sent: {self.frame_count}, Elapsed time: {elapsed_time:.2f} sec"
            )

    def _close(self) -> None:
        for cam in self.cameras.values():
            cam.release()
        for sock in self.sockets.values():
            sock.close()
        self.context.term()
        logger.info("[Image Server] The server has been closed.")

    def _encode_jpeg(self, image):
        if image is None:
            return None
        ret, buffer = cv2.imencode(".jpg", image)
        if not ret:
            return None
        return buffer.tobytes()

    def _encode_depth(self, depth_image):
        if depth_image is None:
            return None
        ret, buffer = cv2.imencode(".png", depth_image)
        if not ret:
            return None
        return buffer.tobytes()

    def _pack_payload(self, payload, timestamp):
        if self.Unit_Test:
            frame_id = self.frame_count
            header = struct.pack("dI", timestamp, frame_id)  # 8-byte double, 4-byte unsigned int
            return header + payload
        return payload

    def send_process(self) -> None:
        colors: dict[str, np.ndarray | None] = defaultdict()
        depths: dict[str, np.ndarray | None] = defaultdict()
        try:
            while True:
                for cam_name, cam in self.cameras.items():
                    img_datas = cam.get_frame()
                    if img_datas is None:
                        logger.error(f"[Image Server] Camera {cam_name} frame read is error.")
                        break
                    colors[cam_name] = img_datas[0]
                    depths[cam_name] = img_datas[1] if len(img_datas) > 1 else None

                timestamp = time.time()

                for cam_name in self.camera_names:
                    parts = []
                    payload = self._encode_jpeg(colors[cam_name])
                    parts.append(self._pack_payload(payload, timestamp))
                    depth_image = depths[cam_name]
                    if depth_image is not None:
                        payload = self._encode_depth(depth_image)
                        parts.append(self._pack_payload(payload, timestamp))
                    if parts:
                        self.sockets[cam_name].send_multipart(parts)
                    else:
                        logger.warning("[Image Server] No frames to publish.")
                        continue

                if self.Unit_Test:
                    current_time = time.time()
                    self._update_performance_metrics(current_time)
                    self._print_performance_metrics(current_time)

        except KeyboardInterrupt:
            logger.warning("[Image Server] Interrupted by user.")
        finally:
            self._close()


if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", help="Path to the camera configuration YAML file.")
    parser.add_argument("--ip", type=str, default="0.0.0.0", help="IP address for the image server.")
    parser.add_argument("--check", action="store_true", help="Check camera connections and exit.")

    args = parser.parse_args()

    try:
        config = {}
        raw_config = yaml.safe_load(open(args.config, "r"))
        for cam_name, cam_cfg in raw_config.items():
            config[cam_name] = CameraConfig(**cam_cfg)
    except Exception as e:
        config = {
            "head_camera": CameraConfig(
                type="opencv", width=640, height=480, fps=30, enable_depth=False, port=5555, idx=4
            )
        }
        logger.warning(f"Using default configuration due to error: {e}")

    server = ImageServer(config=config, ip=args.ip, Unit_Test=args.check)
    server.send_process()

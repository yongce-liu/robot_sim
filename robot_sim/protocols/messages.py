"""ZMQ-based protocol protocol for model interaction."""

import json
import pickle

import numpy as np

try:
    import zmq
except ImportError:
    zmq = None


class ZMQProtocol:
    """ZMQ-based protocol protocol for robot-model interaction.

    Supports both REQ-REP (request-reply) and PUB-SUB (publish-subscribe) patterns.
    """

    def __init__(
        self,
        port: int = 5555,
        host: str = "localhost",
        mode: str = "server",
        pattern: str = "req_rep",
        serialization: str = "json",
    ) -> None:
        """Initialize ZMQ protocol.

        Args:
            port: Port number for protocol
            host: Host address
            mode: "server" or "client"
            pattern: "req_rep" (request-reply) or "pub_sub" (publish-subscribe)
            serialization: "json" or "pickle"
        """
        if zmq is None:
            raise ImportError("pyzmq is not installed. Install with: pip install pyzmq")

        self.port = port
        self.host = host
        self.mode = mode
        self.pattern = pattern
        self.serialization = serialization

        self.context = zmq.Context()
        self.socket = None
        self._setup_socket()

    def _setup_socket(self) -> None:
        """Setup ZMQ socket based on mode and pattern."""
        address = f"tcp://{self.host}:{self.port}"

        if self.pattern == "req_rep":
            if self.mode == "server":
                self.socket = self.context.socket(zmq.REP)
                self.socket.bind(address)
                print(f"[ZMQ Server] listening on {address}")
            else:
                self.socket = self.context.socket(zmq.REQ)
                self.socket.connect(address)
                print(f"[ZMQ Client] Connected to {address}")

        elif self.pattern == "pub_sub":
            if self.mode == "server":
                self.socket = self.context.socket(zmq.PUB)
                self.socket.bind(address)
                print(f"[ZMQ Publisher] Publishing on {address}")
            else:
                self.socket = self.context.socket(zmq.SUB)
                self.socket.connect(address)
                self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
                print(f"[ZMQ Subscriber] Subscribed to {address}")

        else:
            raise ValueError(f"Unknown pattern: {self.pattern}")

    def send(self, data: dict[str, any]) -> None:
        """Send data through ZMQ.

        Args:
            data: dictionary containing data to send
        """
        serialized = self._serialize(data)
        self.socket.send(serialized)

    def receive(self, timeout: int | None = None) -> dict[str, any] | None:
        """Receive data from ZMQ.

        Args:
            timeout: Timeout in milliseconds (None for blocking)

        Returns:
            Received data dictionary or None if timeout
        """
        if timeout is not None:
            self.socket.setsockopt(zmq.RCVTIMEO, timeout)

        try:
            message = self.socket.recv()
            return self._deserialize(message)
        except zmq.Again:
            return None

    def _serialize(self, data: dict[str, any]) -> bytes:
        """Serialize data for transmission."""
        if self.serialization == "json":
            # Convert numpy arrays to lists for JSON
            json_data = self._convert_numpy_to_list(data)
            return json.dumps(json_data).encode("utf-8")
        elif self.serialization == "pickle":
            return pickle.dumps(data)
        else:
            raise ValueError(f"Unknown serialization: {self.serialization}")

    def _deserialize(self, data: bytes) -> dict[str, any]:
        """Deserialize received data."""
        if self.serialization == "json":
            return json.loads(data.decode("utf-8"))
        elif self.serialization == "pickle":
            return pickle.loads(data)
        else:
            raise ValueError(f"Unknown serialization: {self.serialization}")

    def _convert_numpy_to_list(self, data: any) -> any:
        """Recursively convert numpy arrays to lists for JSON."""
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, dict):
            return {k: self._convert_numpy_to_list(v) for k, v in data.items()}
        elif isinstance(data, (list, tuple)):
            return [self._convert_numpy_to_list(item) for item in data]
        return data

    def close(self) -> None:
        """Close ZMQ socket and context."""
        if self.socket is not None:
            self.socket.close()
        self.context.term()
        print("[ZMQ] Connection closed")

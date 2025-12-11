"""Communication protocol implementations."""

from typing import Any


class CommunicationProtocol:
    """Base class for communication protocols."""

    def __init__(self) -> None:
        """Initialize communication protocol."""
        pass

    def send(self, data: Any) -> None:
        """Send data through the communication channel."""
        raise NotImplementedError

    def receive(self) -> Any:
        """Receive data from the communication channel."""
        raise NotImplementedError

    def close(self) -> None:
        """Close the communication channel."""
        raise NotImplementedError

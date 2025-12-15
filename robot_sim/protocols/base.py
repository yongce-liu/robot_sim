from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Message:
    pass


class BaseBridge(ABC):
    """Abstract base class for robot-model communication protocols."""

    @abstractmethod
    def bind(self) -> None:
        """Bind the protocol to start listening for connections."""
        pass

    @abstractmethod
    def send(self, message: str) -> None:
        """Send a message to the robot model."""
        pass

    @abstractmethod
    def receive(self) -> str:
        """Receive a message from the robot model."""
        pass

    @abstractmethod
    def close(self) -> None:
        """Close the protocol connection."""
        pass

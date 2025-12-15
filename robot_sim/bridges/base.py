from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass
class Message:
    topic: str
    """the topic that the socket is subscribing/publishing to"""
    data: Any
    """the any data canbe pickled being sent/received, e.g. array, tensor, dict"""


class BaseBridge(ABC):
    """Abstract base class for robot-model communication protocols."""

    messages: dict[str, Message] = defaultdict(Message)
    """All bridge instances share the same message dictionary."""

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

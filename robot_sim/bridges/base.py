from abc import ABC, abstractmethod
from typing import Any

import msgpack

from robot_sim.backends.types import ActionsType, StatesType


########################### Virtual Classes ##########################
class Message(ABC):
    """A generic message structure for robot communication."""

    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=Message.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=Message.decode_custom_classes)

    @staticmethod
    @abstractmethod
    def encode_custom_classes(obj: Any) -> Any:
        """Custom encoder for msgpack to handle non-standard types."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def decode_custom_classes(obj: Any) -> Any:
        """Custom decoder for msgpack to handle non-standard types."""
        raise NotImplementedError


class BaseInterface(ABC):
    """Abstract base class for robot-model interfaces."""

    state_msg: Message
    action_msg: Message
    """At least exist these two message types."""

    @abstractmethod
    @staticmethod
    def stateArray2stateMsg(state_array: StatesType) -> Any:
        """Convert the StatesType to a structured observation dictionary."""
        raise NotImplementedError

    @abstractmethod
    @staticmethod
    def actionMsg2actionArray(action: Any) -> ActionsType:
        """Convert a structured action dictionary to an array of actions."""
        raise NotImplementedError


############################ Bridge Class ##########################
class BaseBridge(ABC):
    """Abstract base class for robot-model communication protocols."""

    def __init__(self, interface: BaseInterface, *args, **kwargs) -> None:
        self.interface = interface

    @abstractmethod
    def bind(self, *args, **kwargs) -> None:
        """Bind the protocol to start listening for connections."""
        raise NotImplementedError

    @abstractmethod
    def send(self, *args, **kwargs) -> None:
        """Send a message to the robot model."""
        raise NotImplementedError

    @abstractmethod
    def receive(self, *args, **kwargs) -> str:
        """Receive a message from the robot model."""
        raise NotImplementedError

    @abstractmethod
    def close(self, *args, **kwargs) -> None:
        """Close the protocol connection."""
        raise NotImplementedError

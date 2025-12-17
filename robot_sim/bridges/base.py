import io
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import msgpack
import numpy as np
import torch


@dataclass
class Message:
    timestamp: float
    state_array: np.ndarray | torch.Tensor | list[float] | None
    action_array: np.ndarray | torch.Tensor | list[float] | None

    @staticmethod
    def reindex_array(array: np.ndarray | torch.Tensor | list[float], **kwargs) -> Any:
        indices: np.ndarray | torch.Tensor | list[int] = kwargs.get("indices", None)
        if indices is not None:
            return array[..., indices]
        names: list[str] = kwargs.get("names", None)
        if names is not None:
            raise NotImplementedError("Reindexing by names is not implemented yet.")

    def to_tensor(self) -> "Message":
        state_tensor = torch.tensor(self.state_array) if self.state_array is not None else None
        action_tensor = torch.tensor(self.action_array) if self.action_array is not None else None
        return Message(
            timestamp=self.timestamp,
            state_array=state_tensor,
            action_array=action_tensor,
        )

    def to_numpy(self) -> "Message":
        state_array = self.state_array.numpy() if isinstance(self.state_array, torch.Tensor) else self.state_array
        action_array = self.action_array.numpy() if isinstance(self.action_array, torch.Tensor) else self.action_array
        return Message(
            timestamp=self.timestamp,
            state_array=state_array,
            action_array=action_array,
        )


class MessageSerializer:
    @staticmethod
    def to_bytes(data: Any) -> bytes:
        return msgpack.packb(data, default=MessageSerializer.encode_custom_classes)

    @staticmethod
    def from_bytes(data: bytes) -> Any:
        return msgpack.unpackb(data, object_hook=MessageSerializer.decode_custom_classes)

    @staticmethod
    def decode_custom_classes(obj):
        if not isinstance(obj, dict):
            return obj
        if "__ModalityConfig_class__" in obj:
            return ModalityConfig(**obj["as_json"])
        if "__ndarray_class__" in obj:
            return np.load(io.BytesIO(obj["as_npy"]), allow_pickle=False)
        return obj

    @staticmethod
    def encode_custom_classes(obj):
        if isinstance(obj, np.ndarray):
            output = io.BytesIO()
            np.save(output, obj, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        if isinstance(obj, torch.Tensor):
            array = obj.cpu().numpy()
            output = io.BytesIO()
            np.save(output, array, allow_pickle=False)
            return {"__ndarray_class__": True, "as_npy": output.getvalue()}
        return obj


class BaseBridge(ABC):
    """Abstract base class for robot-model communication protocols."""

    messages: dict[str, dict[str, Any]] = defaultdict(dict)
    """All bridge instances share the same message dictionary.
       topicName -> {key: value}
    """

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

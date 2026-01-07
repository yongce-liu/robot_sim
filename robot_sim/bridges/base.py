from abc import ABC, abstractmethod
from typing import Literal

from robot_sim.backends.types import ActionsType, StatesType


class BaseBridge(ABC):
    def __init__(self, mode: Literal["server", "client", "robot"]) -> None:
        self.mode = mode

    @abstractmethod
    def send_state(
        self,
        topic: str,
        state: StatesType,
        tick: int | None = None,
        stamp_ns: int | None = None,
    ) -> None:
        pass

    @abstractmethod
    def get_state(self, topic: str) -> tuple[int, int, StatesType] | None:
        pass

    @abstractmethod
    def send_action(
        self,
        topic: str,
        action: ActionsType,
        tick: int | None = None,
        stamp_ns: int | None = None,
    ) -> None:
        pass

    @abstractmethod
    def get_action(self, topic: str) -> tuple[int, int, ActionsType] | None:
        pass

    @abstractmethod
    def launch(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass


_BRIDGE_REGISTRY: dict[str, type[BaseBridge]] = {}


def bridge_register(*bridges: str):
    norm = [f.lower().lstrip(".") for f in bridges]

    def deco(cls):
        for f in norm:
            if f in _BRIDGE_REGISTRY:
                raise ValueError(f"Duplicate bridge for name: {f}")
            _BRIDGE_REGISTRY[f] = cls
        return cls

    return deco


class BridgeFactory:
    @staticmethod
    def create(type: str, **kwargs) -> BaseBridge:
        if type not in _BRIDGE_REGISTRY:
            raise ValueError(f"Bridge '{type}' is not registered.")
        bridge_cls = _BRIDGE_REGISTRY[type]
        return bridge_cls(**kwargs)

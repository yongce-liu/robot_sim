from abc import ABC, abstractmethod
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import torch

from robot_sim.configs import SensorConfig

if TYPE_CHECKING:
    from robot_sim.backends import BaseBackend


class BaseSensor(ABC):
    """Base class for all sensors."""

    _backend: "BaseBackend | None" = None
    """Backend simulator instance reference. All class share the common backend instance."""

    def __init__(self, config: SensorConfig, **kwargs) -> None:
        ################### private attributes ###################
        self._data: torch.Tensor | np.ndarray | dict[str, torch.Tensor | np.ndarray] | None = None
        """the latest sensor data."""
        self._data_queue: deque[torch.Tensor | np.ndarray] | None = None
        """Current sensor data."""
        self._last_update_cnt_stamp: int = 0
        """Last update count stamp."""
        self._update_interval: int = 1
        """Update interval in simulation steps."""

        self.config = config
        self._data_queue = deque(maxlen=self.config.data_buffer_length)

    def _bind(self, obj_name: str, sensor_name: str, **kwargs) -> None:
        raise NotImplementedError

    def bind(self, backend: "BaseBackend", obj_name: str, sensor_name: str, **kwargs) -> None:
        self._backend = backend
        # Compute update interval based on frequency, if not specified, update every step
        self._update_interval = int(backend.sim_freq / self.config.freq) if self.config.freq is not None else 1
        assert self._update_interval > 0, "Sensor update frequency must be less than or equal to simulation frequency."
        self._bind(obj_name=obj_name, sensor_name=sensor_name, **kwargs)

    def __call__(self, cnt: int, **kwargs) -> torch.Tensor | np.ndarray | None:
        """Update sensor data if frequency allows.

        Args:
            dt: Time delta since last call (in seconds)
            *args, **kwds: Additional arguments for update method
        """
        # cnt: [0, self._backend._sim_freq-1]
        # scenerio: cnt=10 last_cnt=490
        if (cnt - self._last_update_cnt_stamp) % self._update_interval == 0:
            self._update(**kwargs)
            self._data_queue.append(self._data)
            self._last_update_cnt_stamp = cnt
        return self.data

    @abstractmethod
    def _update(self, **kwargs) -> None:
        """Update sensor data from the backend simulator.
        Especially, you only need to update the _data attribute here for different simulator backends.
        """
        raise NotImplementedError

    @property
    def data(self) -> torch.Tensor | np.ndarray | dict[str, torch.Tensor | np.ndarray]:
        """Get the latest sensor data."""
        return self._data

    @property
    def data_queue(self) -> deque[torch.Tensor | np.ndarray | dict[str, torch.Tensor | np.ndarray]]:
        """Get the data queue."""
        return self._data_queue

"""Sensor module for camera, IMU, and other sensors."""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import MISSING
from enum import Enum

import numpy as np
import torch
from loguru import logger

from robot_sim.backends import BaseBackend
from robot_sim.utils import configclass

from .simulator import BackendType


class SensorType(Enum):
    """Enumeration of available sensor types."""

    CAMERA = "camera"
    CONTACT = "contact"


@configclass
class SensorConfig:
    """Base class for all sensors."""

    type: SensorType = MISSING
    """Type of the sensor."""
    freq: float | None = None
    """Update frequency in Hz. It should less than or equal to the simulation frequency."""
    data_buffer_length: int = 1
    """Maximum length of the data queue."""
    ################### private attributes ###################
    _data: torch.Tensor | np.ndarray | None = None
    """the latest sensor data."""
    _data_queue: deque[torch.Tensor | np.ndarray] | None = None
    """Current sensor data."""
    _last_update_cnt_stamp: int = 0
    """Last update count stamp."""
    _update_interval: int = 1
    """Update interval in simulation steps."""
    _backend: "BaseBackend | None" = None
    """Backend simulator instance reference."""


    def _post_init__(self):
        self._data_queue = deque(maxlen=self.data_buffer_length)

    def _bind(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def bind(self, backend: "BaseBackend", *args, **kwargs) -> None:
        self._backend = backend
        # Compute update interval based on frequency, if not specified, update every step
        self._update_interval = int(backend._sim_freq / self.freq) if self.freq is not None else 1
        assert self._update_interval > 0, "Sensor update frequency must be less than or equal to simulation frequency."
        self._bind(*args, **kwargs)

    def __call__(self, cnt: int, *args, **kwargs) -> torch.Tensor | np.ndarray | None:
        """Update sensor data if frequency allows.

        Args:
            dt: Time delta since last call (in seconds)
            *args, **kwds: Additional arguments for update method
        """
        # cnt: [0, self._backend._sim_freq-1]
        # scenerio: cnt=10 last_cnt=490
        if (cnt - self._last_update_cnt_stamp) % self._update_interval == 0:
            self.update(*args, **kwargs)
            self._data_queue.append(self._data)
            self._last_update_cnt_stamp = cnt
        return self.data

    @abstractmethod
    def update(self, *args, **kwargs) -> None:
        """Update sensor data from the backend simulator.
        Especially, you only need to update the _data attribute here for different simulator backends.
        """
        raise NotImplementedError

    @property
    def data(self) -> torch.Tensor | np.ndarray | None:
        """Get the latest sensor data."""
        return self._data

    @property
    def data_queue(self) -> deque[torch.Tensor | np.ndarray] | None:
        """Get the data queue."""
        return self._data_queue

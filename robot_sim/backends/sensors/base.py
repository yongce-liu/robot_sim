from abc import abstractmethod
from collections import deque
from typing import TYPE_CHECKING

import numpy as np
import torch
from loguru import logger

from robot_sim.configs import SensorConfig

if TYPE_CHECKING:
    from robot_sim.backends import BaseBackend


class BaseSensor:
    """Base class for all sensors."""

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

    def __init__(self, config: SensorConfig, **kwargs) -> None:
        self.config = config
        self._data_queue = deque(maxlen=self.config.data_buffer_length)

        # Validate camera configuration
        if self.config.mount_to is not None:
            assert self.config.position is None, "position should not be set when mount_to is specified."
            assert self.config.look_at is None, "look_at should not be set when mount_to is specified."
            # Mounted camera: require mount_to and mount_link
            assert self.config.mount_link is not None, "mount_link must be specified when mount_to is set."
            if self.config.mount_pos is None:
                self.config.mount_pos = [0.0, 0.0, 0.0]
            if self.config.mount_quat is None:
                self.config.mount_quat = [1.0, 0.0, 0.0, 0.0]  # [w, x, y, z]
            logger.info(
                f"Camera will be mounted to '{self.config.mount_to}' at link '{self.config.mount_link}' "
                f"with position {self.config.mount_pos} and quaternion {self.config.mount_quat}."
            )

        else:
            # World frame camera: require pos and look_at
            assert self.config.position is not None, "position must be specified for world frame camera."
            assert self.config.look_at is not None, "look_at must be specified for world frame camera."
            logger.info(f"World frame camera at position {self.config.position} looking at {self.config.look_at}.")

    def _bind(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def bind(self, backend: "BaseBackend", *args, **kwargs) -> None:
        self._backend = backend
        # Compute update interval based on frequency, if not specified, update every step
        self._update_interval = int(backend._sim_freq / self.config.freq) if self.config.freq is not None else 1
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

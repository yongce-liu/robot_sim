"""Gr00t Teleoperation task for Gr00t adapter using Pico Ultra 4 and Motion Tracker."""

# from typing import Any

# import numpy as np
# from loguru import logger

# from robot_sim.adapters.gr00t.env import Gr00tEnv, Gr00tEnvConfig
# from robot_sim.utils.helper import task_register

from robot_sim.adapters.gr00t.env import Gr00tEnv
from robot_sim.utils.helper import task_register


@task_register("Gr00tTeleoperation-v0")
class TeleoperationTask(Gr00tEnv):
    pass

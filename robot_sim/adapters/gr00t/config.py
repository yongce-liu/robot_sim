from dataclasses import MISSING

from robot_sim.configs import SimulatorConfig
from robot_sim.utils.config import configclass


@configclass
class Gr00tConfig:
    """Configuration for Gr00t Experiments.

    Attributes:
        robot_name: Name of the robot (e.g., "g1", "gr1")
        enable_gravity_compensation: Whether to enable gravity compensation
        gravity_compensation_joints: List of joint groups for gravity compensation
    """

    simulator_config: SimulatorConfig = MISSING
    """It can be loaded from a yaml file or defined inline."""

    observation_mapping: dict[str, list[str] | str] = MISSING
    """Mapping of observation groups to joint names, camera, ....; Support regex patterns."""
    action_mapping: dict[str, list[str]] = MISSING
    """Mapping of action groups to joint names; Support regex patterns."""

    allowed_language_charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ,.\n\t[]{}()!?'_:"

    enable_gravity_compensation: bool = False
    gravity_compensation_joints: list[str] = None


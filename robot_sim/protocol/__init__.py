"""Communication module for robot simulation.

This module handles protocol between simulation server and external clients
(VLA models, planners, etc.) using ZMQ protocol.
"""

from robot_sim.protocol.messages import ZMQProtocol
from robot_sim.protocol.protocol import CommunicationProtocol

__all__ = [
    "CommunicationProtocol",
    "ZMQProtocol",
]

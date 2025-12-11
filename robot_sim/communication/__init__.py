"""Communication module for robot simulation.

This module handles communication between simulation server and external clients
(VLA models, planners, etc.) using ZMQ protocol.
"""

from robot_sim.communication.messages import ZMQProtocol
from robot_sim.communication.protocol import CommunicationProtocol

__all__ = [
    "CommunicationProtocol",
    "ZMQProtocol",
]

"""Communication module for robot simulation.

This module handles protocol between simulation server and external clients
(VLA models, planners, etc.) using ZMQ protocol.
"""

from robot_sim.protocols.messages import ZMQProtocol

__all__ = [
    "CommunicationProtocol",
    "ZMQProtocol",
]

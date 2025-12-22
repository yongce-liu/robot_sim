# Import tasks to trigger registration with gym.registry
from . import tasks
from .controller import DecoupledWBCPolicy, InterpolationPolicy, LowerBodyPolicy

__all__ = ["DecoupledWBCPolicy", "LowerBodyPolicy", "InterpolationPolicy"]

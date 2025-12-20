from typing import Any

from robot_sim.controllers import BaseController, CompositeController


class Gr00tWBCPolicy(CompositeController):
    """Whole-body control policy for Gr00t robot.

    This policy combines multiple controllers to produce whole-body commands
    for the Gr00t robot.
    """

    def __init__(self, controllers: dict[str, BaseController]) -> None:
        super().__init__(controllers)

    def compute(self, *args: Any, **kwargs: Any) -> Any:
        # Implement routing logic specific to Gr00t here
        # For example, route high-level commands to low-level PID controllers
        high_level_cmd = self.controllers["high_level"].compute(*args, **kwargs)
        low_level_cmd = self.controllers["pid"].compute(high_level_cmd)
        return low_level_cmd

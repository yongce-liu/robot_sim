from typing import TYPE_CHECKING, Any

from robot_sim.controllers import BaseController, CompositeController

if TYPE_CHECKING:
    from robot_sim.adapters.gr00t.env import Gr00tEnv


class Gr00tWBCPolicy(BaseController):
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


class Gr00tController(CompositeController):
    """Composite controller for Gr00t robot.

    This controller combines multiple sub-controllers to manage different
    aspects of the Gr00t robot's behavior.
    For example, it can include a trained whole-body controller (WBC) and a PD controller.
    In another example, you can implement a unitree-sdk message interface and then use the sdk to control the robot.
    """

    def __init__(self, env: "Gr00tEnv") -> None:
        self.env = env
        controllers = {
            "wbc": Gr00tWBCPolicy(
                {
                    # Initialize sub-controllers here
                    # e.g., "high_level": HighLevelController(env),
                    #       "pid": PIDController(env),
                }
            ),
            # Add other controllers as needed
        }
        super().__init__(controllers)

    def compute(self, action: dict[str, Any]) -> Any:
        # Implement routing logic specific to Gr00t here
        # For example, route commands to different sub-controllers
        results = {}
        for name, controller in self.controllers.items():
            results[name] = controller.compute(action)
        return results

from typing import Any

import numpy as np

from robot_sim.backends.types import ActionsType, ArrayType, StatesType
from robot_sim.controllers import BasePolicy, CompositeController, PIDController


class UpperBodyPolicy(BasePolicy):
    def __init__(self, actuator_indices: list[int]):
        """
        Upper body control policy for Gr00t robot.
        Args:
            actuator_names (list[str], optional): List of actuator names to control, it also defines the actuator order in the action space. If None, defaults to all upper body actuators.
        """
        super().__init__()
        self.actuator_indices = actuator_indices

    def compute(self, states: StatesType, targets: Any) -> ArrayType:
        return targets[..., self.actuator_indices]

    def load_policy(self, policy_path):
        pass

    def reset(self):
        self.actuator_indices = None


class LowerBodyPolicy(BasePolicy):
    def __init__(self, actuator_indices: list[int]):
        """
        Lower body control policy for Gr00t robot.
        Args:
            actuator_names (list[str], optional): List of actuator names to control, it also defines the actuator order in the action space. If None, defaults to all lower body actuators.
        """
        super().__init__()
        self.actuator_indices = actuator_indices

    def compute(self, states: StatesType, targets: Any) -> ArrayType:
        return targets[..., self.actuator_indices]

    def load_policy(self, policy_path):
        pass

    def reset(self):
        self.actuator_indices = None


class DecoupledWBCPolicy(CompositeController):
    """Whole-body control policy for Gr00t robot.

    This policy combines multiple controllers to produce whole-body commands
    for the Gr00t robot.
    """

    def __init__(
        self,
        upper_body_policy: UpperBodyPolicy = None,
        lower_body_policy: LowerBodyPolicy = None,
    ) -> None:
        super().__init__(controllers={"upper_body_policy": upper_body_policy, "lower_body_policy": lower_body_policy})
        self.upper_body_policy = upper_body_policy
        self.lower_body_policy = lower_body_policy

    def compute(self, states: StatesType, targets: Any) -> ActionsType:
        upper_target = self.upper_body_policy.compute(states, targets)
        lower_target = self.lower_body_policy.compute(states, targets)
        output = np.concatenate([upper_target, lower_target], axis=-1)
        return output


class Gr00tWBCController(CompositeController):
    """Composite controller for Gr00t robot.

    This controller combines multiple sub-controllers to manage different
    aspects of the Gr00t robot's behavior.
    For example, it can include a trained whole-body controller (WBC) and a PD controller.
    In another example, you can implement a unitree-sdk message interface and then use the sdk to control the robot.
    """

    def __init__(self, robot_name: str, wbc_policy: BasePolicy, pd_controller: PIDController) -> None:
        super().__init__(controllers={"wbc_policy": wbc_policy, "pd_controller": pd_controller})
        self.robot_name = robot_name
        self.wbc_policy = wbc_policy
        self.pd_controller = pd_controller

    def compute(self, states: StatesType, targets: Any) -> ActionsType:
        # Implement routing logic specific to Gr00t here
        # For example, route commands to different sub-controllers
        output = self.wbc_policy.compute(states=states, targets=targets[self.robot_name])
        # actions = self.pd_controller.compute(
        #     target=output, position=states[self.robot_name].joint_pos, velocity=states[self.robot_name].joint_vel
        # )
        return {self.robot_name: output}

from os import PathLike
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_sim.controllers import BaseController, CompositeController

if TYPE_CHECKING:
    from robot_sim.adapters.gr00t import Gr00tWBCEnv


class InterpolationPolicy(BaseController):
    pass


class LowerBodyPolicy(BaseController):
    pass


class DecoupledWBCPolicy(BaseController):
    """Whole-body control policy for Gr00t robot.

    This policy combines multiple controllers to produce whole-body commands
    for the Gr00t robot.
    """

    def __init__(self, model_path: PathLike, output_indices: list[int] | np.ndarray) -> None:
        pass

    # def compute(self, *args: Any, **kwargs: Any) -> Any:
    #     # Implement routing logic specific to Gr00t here
    #     # For example, route high-level commands to low-level PID controllers
    #     high_level_cmd = self.controllers["high_level"].compute(*args, **kwargs)
    #     low_level_cmd = self.controllers["pid"].compute(high_level_cmd)
    #     return low_level_cmd

    # def reset(self, **kwargs):
    #     obs, info = self.env.reset(**kwargs)
    #     self.wbc_policy = self.setup_wbc_policy()
    #     self.wbc_policy.set_observation(obs)
    #     return obs, info

    # def step(self, action: dict[str, Any]) -> tuple[Any, float, bool, bool, dict[str, Any]]:
    #     action_dict = concat_action(self.robot_model, action)

    #     wbc_goal = {}
    #     for key in ["navigate_cmd", "base_height_command", "target_upper_body_pose"]:
    #         if key in action_dict:
    #             wbc_goal[key] = action_dict[key]

    #     self.wbc_policy.set_goal(wbc_goal)
    #     wbc_action = self.wbc_policy.get_action()

    #     result = super().step(wbc_action)
    #     self.wbc_policy.set_observation(result[0])
    #     return result

    # def setup_wbc_policy(self):
    #     robot_type, robot_model = get_robot_type_and_model(
    #         self.script_config["robot"],
    #         enable_waist_ik=self.script_config.get("enable_waist", False),
    #     )
    #     config = SyncSimDataCollectionConfig.from_dict(self.script_config)
    #     config.update(
    #         {
    #             "save_img_obs": False,
    #             "ik_indicator": False,
    #             "enable_real_device": False,
    #             "replay_data_path": None,
    #         }
    #     )
    #     wbc_config = config.load_wbc_yaml()
    #     wbc_config["upper_body_policy_type"] = "identity"
    #     wbc_policy = get_wbc_policy(robot_type, robot_model, wbc_config, init_time=0.0)
    #     self.total_dofs = len(robot_model.get_joint_group_indices("upper_body"))
    #     wbc_policy.activate_policy()
    #     return wbc_policy

    # def _get_joint_group_size(self, group_name: str) -> int:
    #     """Return the number of joints in a group, cached since lookup is static."""
    #     if not hasattr(self, "_joint_group_size_cache"):
    #         self._joint_group_size_cache = {}
    #     if group_name not in self._joint_group_size_cache:
    #         self._joint_group_size_cache[group_name] = len(self.robot_model.get_joint_group_indices(group_name))
    #     return self._joint_group_size_cache[group_name]

    # def _wbc_action_space(self) -> gym.spaces.dict:
    #     action_space: dict[str, gym.spaces.Space] = {
    #         "action.navigate_command": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
    #         "action.base_height_command": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
    #         "action.left_hand": gym.spaces.Box(
    #             low=-np.inf,
    #             high=np.inf,
    #             shape=(self._get_joint_group_size("left_hand"),),
    #             dtype=np.float32,
    #         ),
    #         "action.right_hand": gym.spaces.Box(
    #             low=-np.inf,
    #             high=np.inf,
    #             shape=(self._get_joint_group_size("right_hand"),),
    #             dtype=np.float32,
    #         ),
    #         "action.left_arm": gym.spaces.Box(
    #             low=-np.inf,
    #             high=np.inf,
    #             shape=(self._get_joint_group_size("left_arm"),),
    #             dtype=np.float32,
    #         ),
    #         "action.right_arm": gym.spaces.Box(
    #             low=-np.inf,
    #             high=np.inf,
    #             shape=(self._get_joint_group_size("right_arm"),),
    #             dtype=np.float32,
    #         ),
    #     }
    #     if (
    #         "waist" in self.robot_model.supplemental_info.joint_groups["upper_body_no_hands"]["groups"]  # type: ignore[attr-defined]
    #     ):
    #         action_space["action.waist"] = gym.spaces.Box(
    #             low=-np.inf,
    #             high=np.inf,
    #             shape=(self._get_joint_group_size("waist"),),
    #             dtype=np.float32,
    #         )
    #     return gym.spaces.dict(action_space)


# def concat_action(robot_model, goal: dict[str, Any]) -> dict[str, Any]:
#     """Combine individual joint-group targets into the upper-body action vector."""
#     processed_goal = {}
#     for key, value in goal.items():
#         processed_goal[key.replace("action.", "")] = value

#     first_value = next(iter(processed_goal.values()))
#     action = np.zeros(first_value.shape[:-1] + (robot_model.num_dofs,))

#     action_dict = {}
#     action_dict["navigate_cmd"] = processed_goal.pop("navigate_command", DEFAULT_NAV_CMD)
#     action_dict["base_height_command"] = np.array(processed_goal.pop("base_height_command", DEFAULT_BASE_HEIGHT))

#     for joint_group, value in processed_goal.items():
#         indices = robot_model.get_joint_group_indices(joint_group)
#         action[..., indices] = value

#     upper_body_indices = robot_model.get_joint_group_indices("upper_body")
#     action = action[..., upper_body_indices]
#     action_dict["target_upper_body_pose"] = action
#     return action_dict


class Gr00tWBCController(CompositeController):
    """Composite controller for Gr00t robot.

    This controller combines multiple sub-controllers to manage different
    aspects of the Gr00t robot's behavior.
    For example, it can include a trained whole-body controller (WBC) and a PD controller.
    In another example, you can implement a unitree-sdk message interface and then use the sdk to control the robot.
    """

    def __init__(self, env: "Gr00tWBCEnv") -> None:
        self.env = env
        controllers = {
            "wbc": DecoupledWBCPolicy(
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

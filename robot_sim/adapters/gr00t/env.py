import regex as re
from loguru import logger

from robot_sim.controllers import PIDController
from robot_sim.envs import MapEnv

from .controller import DecoupledWBCPolicy, Gr00tWBCController, LowerBodyPolicy, UpperBodyPolicy


class Gr00tWBCEnv(MapEnv):
    """Gr00t Whole Body Control Environment.

    This environment uses a composite controller to manage different control strategies for the Gr00t robot.
    The composite controller routes commands to sub-controllers based on the current state and targets.

    Args:
        config (MapEnvConfig): Configuration for the Map environment, including controller settings.
    """

    def _init_controller(
        self,
        actuators_order: list[str] = None,
        upper_actuator_indices: list[int] = None,
        lower_actuator_indices: list[int] = None,
    ) -> Gr00tWBCController:
        """Initialize the composite controller for the Gr00t robot.

        Returns:
            An instance of Gr00tWBCController.
        """
        reindices = self.get_joint_reindices(
            actuators_order=actuators_order,
            upper_actuator_indices=upper_actuator_indices,
            lower_actuator_indices=lower_actuator_indices,
        )

        upper_body_policy = UpperBodyPolicy(actuator_indices=reindices["upper_body"])
        lower_body_policy = LowerBodyPolicy(actuator_indices=reindices["lower_body"])
        wbc_policy = DecoupledWBCPolicy(
            upper_body_policy=upper_body_policy,
            lower_body_policy=lower_body_policy,
        )
        pd_controller = PIDController(kp=100.0, kd=2.0, dt=self.config.simulator_config.sim.dt)
        controller = Gr00tWBCController(
            robot_name=self.robot_name,
            wbc_policy=wbc_policy,
            pd_controller=pd_controller,
        )
        return controller

    def get_joint_reindices(
        self,
        actuators_order: list[str],
        upper_actuator_indices: list[int],
        lower_actuator_indices: list[int],
    ) -> dict[str, list[int]]:
        env_actuator_names = self.get_actuator_names(self.robot_name)
        rx_actuator_names = [re.compile(name) for name in actuators_order]

        default2gr00t = []
        for rx in rx_actuator_names:
            matched_indices = [i for i, env_name in enumerate(env_actuator_names) if rx.fullmatch(env_name)]
            assert len(matched_indices) == 1, (
                f"Expected exactly one actuator to match pattern '{rx}', but found {len(matched_indices)}."
            )
            default2gr00t.extend(matched_indices)

        gr00t2default = []
        for env_name in env_actuator_names:
            matched_inverse_indices = [i for i, rx in enumerate(rx_actuator_names) if rx.fullmatch(env_name)]
            assert len(matched_inverse_indices) == 1, (
                f"Expected exactly one actuator to match '{env_name}', but found {len(matched_inverse_indices)}."
            )
            gr00t2default.extend(matched_inverse_indices)

        # it is upper indices in the actuators_order
        default2gr00t_upper = [default2gr00t[i] for i in upper_actuator_indices]
        default2gr00t_lower = [default2gr00t[i] for i in lower_actuator_indices]

        assert len(gr00t2default) == len(default2gr00t) == len(env_actuator_names), (
            "Expected to match all actuators, but mis-matched."
        )

        # assert check
        # test default2gr00t
        if not all([rx_actuator_names[i].fullmatch(env_actuator_names[idx]) for i, idx in enumerate(default2gr00t)]):
            logger.error(
                f"Default to Gr00t reindexing check failed.\nOriginal: {env_actuator_names}\nReindexed: {[env_actuator_names[idx] for idx in default2gr00t]}\nExpected patterns: {actuators_order}"
            )
        # test gr00t2default
        if not all([rx_actuator_names[idx].fullmatch(env_actuator_names[i]) for i, idx in enumerate(gr00t2default)]):
            logger.error(
                f"Gr00t to Default reindexing check failed.\nOriginal: {env_actuator_names}\nReindexed: {[env_actuator_names[idx] for idx in gr00t2default]}\nExpected patterns: {actuators_order}"
            )
        # test default2gr00t-UpperBody
        if not all(
            [
                rx_actuator_names[rx_idx].fullmatch(env_actuator_names[env_idx])
                for rx_idx, env_idx in zip(upper_actuator_indices, default2gr00t_upper)
            ]
        ):
            logger.error(
                f"Default to Gr00t UpperBody reindexing check failed.\nOriginal: {env_actuator_names}\nReindexed: {[env_actuator_names[idx] for idx in default2gr00t_upper]}\nExpected patterns: {[actuators_order[i] for i in upper_actuator_indices]}"
            )
        # test default2gr00t-LowerBody
        if not all(
            [
                rx_actuator_names[rx_idx].fullmatch(env_actuator_names[env_idx])
                for rx_idx, env_idx in zip(lower_actuator_indices, default2gr00t_lower)
            ]
        ):
            logger.error(
                f"Default to Gr00t LowerBody reindexing check failed.\nOriginal: {env_actuator_names}\nReindexed: {[env_actuator_names[idx] for idx in default2gr00t_lower]}\nExpected patterns: {[actuators_order[i] for i in lower_actuator_indices]}"
            )

        return {
            "whole_body": default2gr00t,
            "upper_body": default2gr00t_upper,
            "lower_body": default2gr00t_lower,
            "whole_body_inverse": gr00t2default,  # concat[upper, lower][inverse]
        }


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

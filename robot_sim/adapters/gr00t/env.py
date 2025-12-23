from typing import Any

import numpy as np
import regex as re

from robot_sim.configs.object import ControlType
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
        target_actuator_names: list[str] = None,
        upper_policy: dict[str, any] = None,
        lower_policy: dict[str, any] = None,
    ) -> Gr00tWBCController:
        """Initialize the composite controller for the Gr00t robot.

        Returns:
            An instance of Gr00tWBCController.
        """

        # reindices = self.get_joint_reindices(
        #     target_actuator_names=target_actuator_names,
        #     upper_actuator_indices=upper_actuator_indices,
        #     lower_actuator_indices=lower_actuator_indices,
        # )
        self.target_actuator_names = target_actuator_names
        self.default_actuator_names = self.get_actuator_names(self.robot_name)
        self.default_joint_names = self.get_joint_names(self.robot_name)

        upper_body_policy = self._init_upper_policy(**upper_policy)
        lower_body_policy = self._init_lower_policy(**lower_policy)

        wbc_policy = self._init_wbc_policy(upper_body_policy=upper_body_policy, lower_body_policy=lower_body_policy)
        pd_controller = self._init_pd_controller()

        controller = Gr00tWBCController(
            wbc_policy=wbc_policy,
            pd_controller=pd_controller,
        )
        return controller

    def _init_upper_policy(self, actuator_indices: list[int]) -> UpperBodyPolicy:
        actuator_indices_in_env = np.array(
            self.get_joint_reindex(
                self.default_actuator_names, [self.target_actuator_names[i] for i in actuator_indices]
            ),
            dtype=np.int32,
        )
        upper_body_policy = UpperBodyPolicy(actuator_indices=actuator_indices_in_env)

        return upper_body_policy

    def _init_lower_policy(
        self,
        actuator_indices: list[int],
        hand_indices: list[int],
        observation_params: dict[str, Any],
        **kwargs,
    ) -> LowerBodyPolicy:
        target_actuator_names = self.target_actuator_names
        actuator_indices_in_env = np.array(
            self.get_joint_reindex(self.default_actuator_names, [target_actuator_names[i] for i in actuator_indices]),
            dtype=np.int32,
        )
        unused_joint_indices_in_env = self.get_joint_reindex(
            self.default_joint_names, [target_actuator_names[i] for i in hand_indices]
        )
        used_joint_indices_in_env = np.array(
            [i for i in range(len(self.default_joint_names)) if i not in unused_joint_indices_in_env], dtype=np.int32
        )
        robot_cfg = self.backend.objects[self.robot_name]
        default_joint_position = np.array(
            [robot_cfg.joints[name].default_position for name in self.default_joint_names], dtype=np.float32
        )

        observation_params["used_joint_indices"] = used_joint_indices_in_env
        observation_params["default_joint_position"] = default_joint_position

        lower_body_policy = LowerBodyPolicy(
            actuator_indices=actuator_indices_in_env,
            observation_params=observation_params,
            **kwargs,
        )
        return lower_body_policy

    def _init_wbc_policy(
        self, upper_body_policy: UpperBodyPolicy, lower_body_policy: LowerBodyPolicy
    ) -> DecoupledWBCPolicy:
        output_indices = self.get_joint_reindex(
            default=self.target_actuator_names,
            target=self.default_actuator_names,
            use_regex=True,
        )
        assert len(output_indices) == len(upper_body_policy.actuator_indices) + len(
            lower_body_policy.actuator_indices
        ), "The total number of output indices must match the sum of upper and lower body policy actuator indices."
        wbc_policy = DecoupledWBCPolicy(
            upper_body_policy=upper_body_policy,
            lower_body_policy=lower_body_policy,
            output_redices=output_indices,
        )
        return wbc_policy

    def _init_pd_controller(self) -> PIDController:
        robot_cfg = self.backend.objects[self.robot_name]
        kp = np.array([robot_cfg.joints[name].stiffness for name in self.default_actuator_names], dtype=np.float32)
        kd = np.array([robot_cfg.joints[name].damping for name in self.default_actuator_names], dtype=np.float32)
        used_pd_indices = [
            i
            for i, name in enumerate(self.default_actuator_names)
            if ControlType(robot_cfg.joints[name].control_type) == ControlType.TORQUE
        ]
        # kp = np.array([150, 150, 150, 300,  40,  40, 150, 150, 150, 200,  40,  40, 250., 250., 250., 100, 100,  40,  40,  20,  20,  20, 0,0,0,0,0,0,0,100, 100,  40,  40,  20,  20,  20 ,0,0,0,0,0,0,0,], dtype=np.float32)
        # kd = np.array([2, 2, 2, 4, 2, 2, 2, 2, 2, 4, 2, 2, 5., 5., 5., 5., 5., 2., 2., 2., 2., 2.,0,0,0,0,0,0,0,5., 5., 2., 2., 2., 2., 2.,0,0,0,0,0,0,0,], dtype=np.float32)
        pd_controller = PIDController(kp=kp, kd=kd, dt=self.config.simulator_config.sim.dt, enabled_indices=used_pd_indices)

        return pd_controller

    def get_joint_reindex(self, default: list[str], target: list[str], use_regex: bool = True) -> list[int]:
        if use_regex:
            default = [re.compile(name) for name in default]
        reindices = []
        for name in target:
            if use_regex:
                matched_indices = [i for i, rx in enumerate(default) if rx.fullmatch(name)]
            else:
                matched_indices = [i for i, env_name in enumerate(default) if env_name == name]
            assert len(matched_indices) == 1, (
                f"Expected exactly one match for actuator '{name}', but found {len(matched_indices)}."
            )
            reindices.append(matched_indices[0])

        return reindices

    def check_joint_reindex(
        self,
        default: list[str],
        target: list[str],
        reindices: list[int],
        use_regex: bool = True,
    ) -> None:
        if use_regex:
            default = [re.compile(name) for name in default]
        for i, idx in enumerate(reindices):
            name = target[i]
            if use_regex:
                assert default[idx].fullmatch(name), f"Reindexing check failed for actuator '{name}' at index {idx}."
            else:
                assert default[idx] == name, f"Reindexing check failed for actuator '{name}' at index {idx}."

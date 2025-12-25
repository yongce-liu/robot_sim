import os
from collections import deque
from functools import partial
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from loguru import logger

import robot_sim
from robot_sim.adapters.gr00t.utils import rpy_cmd_from_waist
from robot_sim.backends.types import ActionsType, ArrayType, StatesType
from robot_sim.controllers import BasePolicy, CompositeController, PIDController
from robot_sim.utils.math_array import quat_apply_inverse


class UpperBodyPolicy(BasePolicy):
    def __init__(self, actuator_indices: list[int]):
        """
        Upper body control policy for Gr00t robot.
        Args:
            actuator_names (list[str], optional): List of actuator names to control, it also defines the actuator order in the action space. If None, defaults to all upper body actuators.
        """
        self.actuator_indices = actuator_indices

    def compute(self, name: str, states: StatesType, targets: ActionsType, **kwargs) -> ArrayType:
        return targets[name][..., self.actuator_indices]

    def load_policy(self, policy_path):
        pass

    def reset(self):
        self.actuator_indices = None


class LowerBodyPolicy(BasePolicy):
    def __init__(
        self,
        actuator_indices: list[int],
        model_path: dict[Literal["stand", "walk"], str] = None,
        observation_params: dict[str, Any] | None = None,
        stand_mode_threshold: float = 0.05,
        use_rpy_cmd_from_waist: bool = True,
        **kwargs,
    ):
        """
        Lower body control policy for Gr00t robot.
        Args:
            actuator_names (list[str], optional): List of actuator names to control, it also defines the actuator order in the action space. If None, defaults to all lower body actuators.
        """
        self.actuator_indices = actuator_indices
        self.model_path = model_path
        self.stand_mode_threshold = stand_mode_threshold

        project_root = Path(robot_sim.__file__).parents[1].resolve()
        self.policies = {}
        if self.model_path is not None:
            for k, v in self.model_path.items():
                path = project_root / Path(v)
                self.policies[k] = self.load_onnx_policy(path)
            self.init_by_config(observation_params)
        if use_rpy_cmd_from_waist:
            self.rpy_cmd_from_waist = partial(
                rpy_cmd_from_waist, torso_index=kwargs["torso_index"], pelvis_index=kwargs["pelvis_index"], action=None
            )
        else:
            self.rpy_cmd_from_waist = None

    def compute(self, name: str, states: StatesType, targets: ActionsType) -> ArrayType:
        if self.model_path is None:
            return targets[name][..., self.actuator_indices]

        with torch.no_grad():
            self._obs_buffer.append(self.compose_observation(name, states, targets))
            nav_cmd = targets.get("navigate_command", np.zeros((1, 3)))
            if np.linalg.norm(nav_cmd) < self.stand_mode_threshold:
                policy = self.policies["stand"]
            else:
                policy = self.policies["walk"]

            self.action = policy(self.observation).detach().cpu().numpy()
            cmd_q = self.action * self.action_scale + self.actuator_default_joint_position

            return cmd_q

    def load_policy(self, path: os.PathLike) -> None:
        return self.load_onnx_policy(path)

    def load_onnx_policy(self, path: os.PathLike):
        import onnxruntime as ort
        import torch

        print(f"Loading ONNX policy from {path}")
        model = ort.InferenceSession(path)

        def run_inference(input_tensor):
            if isinstance(input_tensor, torch.Tensor):
                input_np = input_tensor.detach().cpu().numpy()
            else:
                input_np = np.asarray(input_tensor)
            ort_inputs = {model.get_inputs()[0].name: input_np}
            ort_outs = model.run(None, ort_inputs)
            return torch.tensor(ort_outs[0], device="cpu")

        logger.info(f"Successfully loaded ONNX policy from {path}")

        return run_inference

    def compose_observation(self, name: str, states: StatesType, targets: dict[str, Any]) -> torch.Tensor:
        quat = states[name].root_state[..., 3:7]  # [w, x, y, z]
        omega = states[name].root_state[..., 10:13]  # angular velocity in world frame

        nav_cmd = targets.get("action.navigate_command", self.default_nav_command) * self.command_scale
        height_cmd = np.array([targets.get("action.base_height_command", self.default_height_cmd)])
        rpy_cmd = (
            self.rpy_cmd_from_waist(name=name, states=states)
            if self.rpy_cmd_from_waist is not None
            else self.default_rpy_cmd
        )
        omega_scaled = omega * self.ang_vel_scale
        gravity_orientation = quat_apply_inverse(quat, self.gravity_vec)
        joint_pos_scaled = (states[name].joint_pos - self.default_joint_position) * self.joint_pos_scale
        joint_vel_scaled = states[name].joint_vel * self.joint_vel_scale
        action_prev = self.action

        obs = np.concatenate(
            [
                nav_cmd,
                height_cmd,
                rpy_cmd,
                omega_scaled,
                gravity_orientation,
                joint_pos_scaled[..., self.used_joint_indices],
                joint_vel_scaled[..., self.used_joint_indices],
                action_prev,
            ],
            axis=-1,
            dtype=np.float32,
        )

        return torch.from_numpy(obs)

    def init_by_config(self, config: dict[str, Any]) -> None:
        self.obs_cfg = config
        self.gravity_vec = np.array(self.obs_cfg.get("gravity_vec", [0, 0, -1.0]), dtype=np.float32)[np.newaxis, :]
        self.single_obs_dim = self.obs_cfg["single_obs_dim"]
        self.obs_history_len = self.obs_cfg["obs_history_len"]
        self.action_scale = self.obs_cfg["action_scale"]
        self.command_scale = np.array(self.obs_cfg["command_scale"], dtype=np.float32)[np.newaxis, :]
        self.default_nav_command = np.array(self.obs_cfg["nav_command"], dtype=np.float32)[np.newaxis, :]
        self.default_height_cmd = self.obs_cfg["height_command"]
        self.default_rpy_cmd = np.array(self.obs_cfg["rpy_command"], dtype=np.float32)[np.newaxis, :]
        self.ang_vel_scale = self.obs_cfg["ang_vel_scale"]
        self.joint_pos_scale = self.obs_cfg["joint_pos_scale"]
        self.joint_vel_scale = self.obs_cfg["joint_vel_scale"]

        self.default_joint_position = self.obs_cfg["default_joint_position"]
        self.used_joint_indices = self.obs_cfg["used_joint_indices"]
        self.actuator_default_joint_position = self.default_joint_position[..., self.actuator_indices]
        self._obs_buffer = deque(
            [torch.zeros(size=(1, self.single_obs_dim), dtype=torch.float32) for _ in range(self.single_obs_dim)],
            maxlen=self.obs_history_len,
        )
        self.action = np.zeros((1, len(self.actuator_indices)), dtype=np.float32)

    @property
    def observation(self) -> torch.Tensor:
        return torch.concatenate(list(self._obs_buffer), dim=-1)

    def reset(self):
        pass


class DecoupledWBCPolicy(CompositeController):
    """Whole-body control policy for Gr00t robot.

    This policy combines multiple controllers to produce whole-body commands
    for the Gr00t robot.
    """

    def __init__(
        self,
        upper_body_policy: UpperBodyPolicy = None,
        lower_body_policy: LowerBodyPolicy = None,
        output_indices: list[int] = None,
        lower_priority: bool = True,
    ) -> None:
        super().__init__(controllers={"upper_body_policy": upper_body_policy, "lower_body_policy": lower_body_policy})
        self.upper_body_policy = upper_body_policy
        self.lower_body_policy = lower_body_policy
        self.lower_priority = lower_priority
        self.output_indices = output_indices

    def compute(self, name: str, states: StatesType, targets: ActionsType) -> ActionsType:
        upper_target = self.upper_body_policy.compute(name=name, states=states, targets=targets)
        lower_target = self.lower_body_policy.compute(name=name, states=states, targets=targets)

        if self.lower_priority:
            output = np.concatenate([lower_target, upper_target], axis=-1)
        else:
            output = np.concatenate([upper_target, lower_target], axis=-1)
        return output[..., self.output_indices]


class Gr00tWBCController(CompositeController):
    """Composite controller for Gr00t robot.

    This controller combines multiple sub-controllers to manage different
    aspects of the Gr00t robot's behavior.
    For example, it can include a trained whole-body controller (WBC) and a PD controller.
    In another example, you can implement a unitree-sdk message interface and then use the sdk to control the robot.
    """

    def __init__(self, wbc_policy: BasePolicy, pd_controller: PIDController) -> None:
        super().__init__(controllers={"wbc_policy": wbc_policy, "pd_controller": pd_controller})
        self.wbc_policy = wbc_policy
        self.pd_controller = pd_controller

    def compute(self, name: str, states: StatesType, targets: ActionsType) -> ActionsType:
        # Implement routing logic specific to Gr00t here
        # For example, route commands to different sub-controllers
        # wbc_output = default order of actuators
        wbc_output = self.wbc_policy.compute(name=name, states=states, targets=targets)
        pd_output = self.pd_controller.compute(
            target=wbc_output, position=states[name].joint_pos, velocity=states[name].joint_vel
        )
        return pd_output

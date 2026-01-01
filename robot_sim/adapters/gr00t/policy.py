import os
from collections import deque
from functools import partial
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch
from loguru import logger

import robot_sim
import robot_sim.utils.math_array as rmath
from robot_sim.backends.types import ActionsType, ArrayType, ObjectState, StatesType
from robot_sim.utils.math_array import quat_apply_inverse


def rpy_cmd_from_waist(torso_index: int, pelvis_index: int, state: ObjectState) -> ArrayType:
    # Compute torso orientation relative to waist, to pass to lower body policy
    body_state = state.body_state
    #################### Previouos ######################################
    # torso_orientation = matrix_from_quat(body_state[..., torso_index, 3:7])
    # waist_orientation = matrix_from_quat(body_state[..., pelvis_index, 3:7])
    # # Extract yaw from rotation matrix and create a rotation with only yaw
    # # The rotation property is a 3x3 numpy array
    # waist_yaw = np.arctan2(waist_orientation[1, 0], waist_orientation[0, 0])
    # # Create a rotation matrix with only yaw using Pinocchio's rpy functions
    # waist_yaw_only_rotation = matrix_from_euler([0, 0, float(waist_yaw)])
    # yaw_only_waist_from_torso = waist_yaw_only_rotation.T @ torso_orientation
    # torso_orientation_rpy = rotmat_to_rpy(yaw_only_waist_from_torso)
    #################### Current ######################################
    torso_quat = body_state[..., torso_index, 3:7]  # (B, 4), (w, x, y, z)
    waist_quat = body_state[..., pelvis_index, 3:7]  # (B, 4)
    waist_yaw_quat = rmath.yaw_quat(waist_quat)  # (B, 4)
    waist_yaw_quat_inv = rmath.quat_conjugate(waist_yaw_quat)  # (B, 4)
    yaw_only_waist_from_torso_quat = rmath.quat_mul(waist_yaw_quat_inv, torso_quat)
    rpy_cmd = np.stack(rmath.euler_xyz_from_quat(yaw_only_waist_from_torso_quat), axis=-1)

    return rpy_cmd


class UpperBodyPolicy:
    def __init__(self, actuator_indices: list[int]):
        """
        Upper body control policy for Gr00t robot.
        Args:
            actuator_names (list[str], optional): List of actuator names to control, it also defines the actuator order in the action space. If None, defaults to all upper body actuators.
        """
        self.actuator_indices = actuator_indices

    def get_target(self, target: ArrayType, **kwargs) -> ArrayType:
        return target[..., self.actuator_indices]


class LowerBodyPolicy:
    def __init__(
        self,
        actuator_indices: list[int] | ArrayType,
        model_path: dict[Literal["stand", "walk"], str] | None = None,
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
            if observation_params is not None:
                self.init_by_config(observation_params)
        if use_rpy_cmd_from_waist:
            self.rpy_cmd_from_waist = partial(
                rpy_cmd_from_waist, torso_index=kwargs["torso_index"], pelvis_index=kwargs["pelvis_index"]
            )
        else:
            self.rpy_cmd_from_waist = lambda state: None

    def get_target(
        self,
        state: ObjectState,
        target: ArrayType,
        nav_cmd: ArrayType | None = None,
        height_cmd: ArrayType | None = None,
        rpy_cmd: ArrayType | None = None,
    ) -> ArrayType:
        if self.model_path is None:
            return target[..., self.actuator_indices]

        nav_cmd = nav_cmd if nav_cmd is not None else self.default_nav_command
        height_cmd = height_cmd if height_cmd is not None else self.default_height_cmd
        rpy_cmd = rpy_cmd if rpy_cmd is not None else self.rpy_cmd_from_waist(state=state)

        with torch.no_grad():
            if np.linalg.norm(nav_cmd) < self.stand_mode_threshold:
                policy = self.policies["stand"]
            else:
                policy = self.policies["walk"]
            self._obs_buffer.append(
                self.compose_observation(
                    state=state, target=target, nav_cmd=nav_cmd, height_cmd=height_cmd, rpy_cmd=rpy_cmd
                )
            )
            self.action = policy(self.observation).detach().cpu().numpy()
            cmd_q: np.ndarray = self.action * self.action_scale + self.actuator_default_joint_position

            return cmd_q

    def load_policy(self, path: os.PathLike) -> Callable:
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

    def compose_observation(
        self, state: ObjectState, target: ArrayType, nav_cmd: ArrayType, height_cmd: ArrayType, rpy_cmd: ArrayType
    ) -> torch.Tensor:
        quat = state.root_state[..., 3:7]  # [w, x, y, z]
        omega = state.root_state[..., 10:13]  # angular velocity in world frame

        # nav_cmd = targets.get("action.navigate_command", self.default_nav_command * self.command_scale
        # height_cmd = np.array([targets.get("action.base_height_command", self.default_height_cmd)])
        # rpy_cmd = self.rpy_cmd_from_waist(state=state) if self.rpy_cmd_from_waist is not None else self.default_rpy_cmd
        omega_scaled = omega * self.ang_vel_scale
        gravity_orientation = quat_apply_inverse(quat, self.gravity_vec)
        joint_pos_scaled = (state.joint_pos - self.default_joint_position) * self.joint_pos_scale
        joint_vel_scaled = state.joint_vel * self.joint_vel_scale
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
        self.default_height_cmd = np.array([self.obs_cfg["height_command"]], dtype=np.float32)[np.newaxis, :]
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


class DecoupledWBCPolicy:
    """Whole-body control policy for Gr00t robot.

    This policy combines multiple controllers to produce whole-body commands
    for the Gr00t robot.
    """

    def __init__(
        self,
        upper_body_policy: UpperBodyPolicy,
        lower_body_policy: LowerBodyPolicy,
        output_clips: tuple[ArrayType, ArrayType] | None = None,
        output_indices: ArrayType | list[int] | None = None,
        lower_priority: bool = True,
    ) -> None:
        self.upper_body_policy = upper_body_policy
        self.lower_body_policy = lower_body_policy
        self.lower_priority = lower_priority
        self.output_indices = output_indices
        self.output_clips = output_clips

    def __call__(self, name: str, states: StatesType, action: ActionsType) -> ArrayType:
        upper_target = self.upper_body_policy.get_target(state=states[name], target=action[name])
        lower_target = self.lower_body_policy.get_target(
            state=states[name],
            target=action[name],
            nav_cmd=action.get("navigate_command"),
            height_cmd=action.get("base_height_command"),
            rpy_cmd=action.get("rpy_command"),
        )

        if self.lower_priority:
            output: ArrayType = np.concatenate([lower_target, upper_target], axis=-1)
        else:
            output: ArrayType = np.concatenate([upper_target, lower_target], axis=-1)

        if self.output_indices is not None:
            output = output[..., self.output_indices]
        if self.output_clips is not None:
            output = output.clip(self.output_clips[0], self.output_clips[1])

        return output

from typing import Literal

from robot_sim.configs.types import ActionsType, ArrayType, StatesType


def obs_joint_extract(
    name: str,
    states: StatesType,
    indices: list[int],
    mode: Literal["position", "torque"] = "position",
) -> ArrayType:
    robot_state = states[name]
    if mode == "torque":
        return robot_state.joint_action[0, indices]
    elif mode == "position":
        return robot_state.joint_pos[0, indices]
    else:
        raise ValueError(f"Unsupported mode '{mode}' for joint_map. Available modes are 'position' and 'torque'.")


def act_joint_assign(
    name: str,
    group_name: str,
    indices: list[int],
    action: ActionsType,
    **kwargs,
):
    action[name][..., indices] = action[group_name]

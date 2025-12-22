from typing import Any


class Gr00tInterface:
    robot_name: str = "g1"
    message: Any

    def stateArray2stateMsg(self) -> dict:
        """Convert the StatesType to a structured observation dictionary.
        It denpends on the definition/structure of the observation space of Gr00t robot.
        """
        return {
            "joint_positions": self.joint_positions,
            "joint_velocities": self.joint_velocities,
            "motor_torques": self.motor_torques,
            "end_effector_position": self.end_effector_position,
            "end_effector_orientation": self.end_effector_orientation,
            "camera_image": self.camera_image,
        }

    def actionMsg2actionArray(self, action: dict) -> list[float]:
        return action.get("joint_commands", [])

    def msg2bytes(self) -> bytes:
        return self.message.to_bytes()

    def bytes2msg(self, data: bytes) -> Any:
        return self.message.from_bytes(data)

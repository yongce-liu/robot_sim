import os
import tempfile
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
from dm_control import mjcf
from loguru import logger

from robot_sim.backends.base import BaseBackend
from robot_sim.backends.sensors import _SENSOR_TYPE_REGISTRY
from robot_sim.backends.types import ActionsType, ArrayType, ObjectState, StatesType
from robot_sim.configs import ObjectConfig, ObjectType, SimulatorConfig


class MujocoBackend(BaseBackend):
    def __init__(self, config: SimulatorConfig, optional_queries: dict[str, Any] | None = None):
        super().__init__(config, optional_queries)
        self._actions_cache: ActionsType = {}  # robot: action
        assert self.num_envs == 1, f"Mujoco only supports single env, got {self.num_envs}."
        assert self.device == "cpu", f"Mujoco only supports CPU device, got {self.device}."

        # self._mjcf_sub_models: dict[str, mjcf.RootElement] = {}  # robot/object name -> mjcf model
        self._mjcf_model: mjcf.RootElement | None = None
        self._mjcf_physics: mjcf.Physics | None = None
        self.viewer: mujoco.viewer.Viewer | None = None
        # self._mujoco_robot_names = []
        # self._robot_num_dofs = []
        # self._robot_paths = []
        # self._gravity_compensations = []

        # # Support multiple robots
        # for robot in self.robots:
        #     self._robot_paths.append(robot.mjcf_path)
        #     self._gravity_compensations.append(not robot.enabled_gravity)

        # self.viewer = None
        # self.cameras = []
        # # for camera in config.cameras:
        # #     self.cameras.append(camera)
        # self._current_action = None
        # self._current_vel_target = None  # Track velocity targets

        # # === Added: GL/physics serialization + native renderer state ===
        # self._mj_lock = threading.RLock()
        # self._mj_model = None  # native mujoco.MjModel for offscreen rendering
        # self._mj_data = None  # native mujoco.MjData  for offscreen rendering
        # self.renderer = None  # mujoco.Renderer (offscreen)

    def _init_backend(self) -> mjcf.RootElement:
        if self.config.scene.path is not None:
            mjcf_model = mjcf.from_path(self.config.scene.path)
            logger.info(f"Loaded scene from: {self.config.scene.path}")
        else:
            mjcf_model = mjcf.RootElement()
            self._add_terrain(mjcf_model)

        # self._add_cameras(model)
        self._add_objects(mjcf_model)
        # dt
        mjcf_model.option.timestep = self.cfg_phyx.dt
        # gravity
        mjcf_model.option.gravity = self.cfg_phyx.gravity

        self._mjcf_model = mjcf_model
        self._update_buffer_dict(mjcf_model)  # update the root model indices first

    def _launch(self) -> None:
        """Initialize MuJoCo model with optional scene support."""

        assert self._mjcf_model is not None, "MuJoCo model must be initialized."
        self._mjcf_physics = mjcf.Physics.from_mjcf_model(self._mjcf_model)

        # Export MJCF + assets to a temp dir.
        # Handle filename variability (dm_control 1.0.34).
        self.export_mjcf(model=self._mjcf_model, out_dir=tempfile.gettempdir(), file_name="model.xml")

        # FIXME: whether need to reload the model?
        # # Load the model from the XML *in the same directory as the exported assets*
        # # so hashed filenames resolve correctly.
        # self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
        # self._mj_data = mujoco.MjData(self._mj_model)

        # # Create a default-sized renderer (camera sizes can be applied on demand)
        # self.renderer = mujoco.Renderer(self._mj_model, width=640, height=480)

        if not self.headless:
            self.viewer = mujoco.viewer.launch_passive(self._mjcf_physics.model.ptr, self._mjcf_physics.data.ptr)
            self.viewer.sync()

    def _render(self) -> None:
        if self.viewer is not None:
            self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()

    def _simulate(self):
        # # Apply gravity compensation for all robots
        # for robot_name, robot_cfg in enumerate(self.robots):
        #     if self._gravity_compensations[robot_name]:
        #         self._disable_robotgravity()

        self._mjcf_physics.step()

    def _set_states(self, states: StatesType, env_ids: ArrayType | None = None):
        """Set the states of all objects and robots."""
        if env_ids is None:
            env_ids = self._full_env_ids
        for obj_name, obj_state in states.items():
            self._set_root_state(obj_name, obj_state, env_ids)
            self._set_joint_state(obj_name, obj_state, env_ids)
        self._mjcf_physics.forward()

    def _set_actions(self, actions: ActionsType, env_ids: ArrayType | None = None) -> None:
        """Set actions for all robots/objects with ctrl entrypoint."""
        self._actions_cache = actions  # dict[str, np.ndarray (num_envs, num_dofs)]

        if env_ids is None:
            env_ids = self._full_env_ids

        for obj_name, obj_action in actions.items():
            action_indices = self._get_action_indices(obj_name)
            # Here, we also use the default order index when adding a object/robot
            self._mjcf_physics.data.ctrl[action_indices] = obj_action[env_ids]

    def _get_states(self, dtype=np.float32) -> StatesType:
        """Get states of all objects and robots."""

        states: dict[str, ObjectState] = {}
        _pnd = self._mjcf_physics.named.data
        for obj_name in self._buffer_dict.keys():
            joint_names = self.get_joint_names(obj_name)
            _root_state, _body_state = self._pack_state(obj_name)
            state = ObjectState(
                root_state=_root_state.astype(dtype),
                body_state=_body_state.astype(dtype),
                joint_pos=None if len(joint_names) == 0 else _pnd.qpos[joint_names].copy()[None, ...].astype(dtype),
                joint_vel=None if len(joint_names) == 0 else _pnd.qvel[joint_names].copy()[None, ...].astype(dtype),
                joint_action=None,
                sensors={k: v.data for k, v in self._buffer_dict[obj_name].sensors.items()},
            )
            states[obj_name] = state

        return states

    def _pack_state(self, obj_name: str):
        """
        Pack pos(3), quat(4), lin_vel_world(3), ang_vel(3) for one-env MuJoCo.

        Args:
            obj_name: name of the object

        Returns:
            root_np: numpy (13,)      — the first body
            body_np: numpy (n_body,13)     — n_body bodies
        """
        data = self._mjcf_physics.named.data
        body_names = [obj_name + "/"] + self.get_body_names(obj_name)
        pos = data.xpos[body_names]
        quat = data.xquat[body_names]

        # angular ω (world) & v @ subtree_com
        w = data.cvel[body_names, 0:3]
        v = data.cvel[body_names, 3:6]
        # compute world‐frame linear velocity at body origin
        offset = data.xpos[body_names] - data.subtree_com[body_names]
        lin_world = v + np.cross(w, offset)

        full = np.concatenate([pos, quat, lin_world, w], axis=1)
        root_np = full[0]
        return root_np[None, ...], full[1:][None, ...]  # root, bodies

    # Private methods for initializing MuJoCo model
    def _add_terrain(self, model: mjcf.RootElement) -> None:
        """Add default ground plane."""
        if self.terrain is None:
            logger.info("Using terrain from the scene file.")
            return
        else:
            if self.terrain.type == "plane":
                model.asset.add(
                    "texture",
                    name="texplane",
                    type="2d",
                    builtin="checker",
                    mark="edge",
                    width=512,
                    height=512,
                    rgb1=[0.2, 0.3, 0.4],
                    rgb2=[0.1, 0.2, 0.3],
                    markrgb=[0.8, 0.8, 0.8],
                )
                model.asset.add(
                    "material",
                    name="matplane",
                    texture="texplane",
                    texrepeat=[2, 2],
                    texuniform=True,
                    reflectance="0.2",
                    specular="0.2",
                    shininess="0.4",
                    emission="0.05",
                )
                model.worldbody.add(
                    "geom",
                    type="plane",
                    pos="0 0 0",
                    size="100 100 0.001",
                    quat="1 0 0 0",
                    condim="3",
                    conaffinity="15",
                    material="matplane",
                    friction="1.0 0.005 0.0001",  # friction coefficients [sliding, torsional, rolling]
                    solimp="0.9 0.95 0.001 0.5 2",  # restition parameters [min, max, margin, stiffness, damping]
                    solref="0.02 1",  # contact stiffness and damping [timeconst, dampratio]
                )
                # FIXME: Temporary headlight settings (dm_control default is too low)
                model.visual.headlight.diffuse = [0.6, 0.6, 0.6]
                model.visual.headlight.ambient = [0.3, 0.3, 0.3]
                model.visual.headlight.specular = [0.0, 0.0, 0.0]
                model.visual.rgba.haze = [0.15, 0.25, 0.35, 1.0]
            else:
                raise NotImplementedError(f"Terrain type '{self.terrain.type}' not supported yet in MuJoCo backend.")

    def _add_objects(self, model: mjcf.RootElement) -> None:
        """Add individual objects to the model."""
        for obj_name, obj_cfg in self.objects.items():
            if obj_cfg.type in [ObjectType.CUSTOM, ObjectType.ROBOT]:
                obj_mjcf = mjcf.from_path(obj_cfg.path)
                # TODO: handle free joints for custom objects
                """
                # Remove free joint since dm_control has limit support for it.
                for joint in obj_mjcf.find_all("joint"):
                    if joint.tag == "joint" and joint.type == "free":
                        joint.remove()
                """
            else:
                xml_str = self._create_builtin_xml(obj_cfg)
                obj_mjcf = mjcf.from_xml_string(xml_str)

            obj_mjcf.model = obj_name
            obj_attached = model.attach(obj_mjcf)

            # FIXME: temporary fix for free joint handling in dm_control
            self._fix_attach_bugs(obj_attached, obj_name, obj_cfg)

            # Apply default position and orientation to the object's root body,
            obj_attached.pos = obj_cfg.initial_position
            obj_attached.quat = obj_cfg.initial_orientation  # [w,x,y,z] format

            # self._mjcf_sub_models[obj_name] = obj_mjcf
            # self._update_buffer_dict(obj_mjcf, obj_cfg)

    @staticmethod
    def _fix_attach_bugs(obj_attached: mjcf.RootElement, obj_name: str, obj_cfg: ObjectConfig) -> None:
        """
        # FIXME: A temporary workaround for free joint handling in dm_control
        temporaly fix attaching robots bugs in dm_control when attach a object.
        """

        all_joints = obj_attached.find_all("joint")
        if len(all_joints) > 0:
            child_joint = all_joints[0]
            if child_joint.type == "free":
                logger.warning(
                    f"Robot '{obj_name}' already has a free joint in its MJCF. "
                    "Use dm_control load the object will create a dummy body. "
                    "We will remove it and add the free joint at the root body level."
                )
                child_joint.remove()
        if not obj_cfg.properties.get("fix_base_link", False):  # default: False, assume the robot can move freely
            obj_attached.add("freejoint")

        # FIXME: Ensure the attached robot has an inertial element to avoid simulation issues, (height)
        all_bodies = obj_attached.find_all("body")
        if len(all_bodies) > 0 and not hasattr(obj_attached, "inertial") or obj_attached.inertial is None:
            child_body = all_bodies[0]  # The first child body after the root
            if child_body.pos is not None:
                obj_attached.pos = child_body.pos
                child_body.pos = "0 0 0"  # Reset child body position to origin with respect to the attached robot
            if child_body.quat is not None:
                obj_attached.quat = child_body.quat
                child_body.quat = "1 0 0 0"
            # pos = child_body.inertial.pos
            # obj_attached.add("inertial", mass="1e-9", diaginertia="1e-9 1e-9 1e-9", pos=pos)

    @staticmethod
    def export_mjcf(model: mjcf.RootElement, out_dir: os.PathLike, file_name: str = "model.xml") -> None:
        """Export the full MJCF model and assets to the specified directory."""
        # Write assets + XML to disk (this version returns None and writes files)
        # model_name is not guaranteed to be respected by all versions, so we’ll glob later.
        mjcf.export_with_assets(model, out_dir, out_file_name=file_name)
        logger.info(f"Exported MJCF model and assets to: {out_dir}/{file_name}")

    def _update_buffer_dict(self, model: mjcf.RootElement) -> None:
        """Update joint and body name indices for the given model."""
        all_joints = model.find_all("joint")
        all_bodies = model.find_all("body")
        actuator_names = [actuator.full_identifier for actuator in model.find_all("actuator")]

        for obj_name, obj_cfg in self.objects.items():
            # Only find joints and bodies belonging to this specific object
            # In dm_control, attached elements are prefixed with the model name
            obj_prefix = f"{obj_name}/"

            for joint in all_joints:
                # Check if joint belongs to this object by checking the full_identifier
                full_identifier = joint.full_identifier
                if full_identifier and full_identifier.startswith(obj_prefix):
                    # Extract the joint name without the prefix
                    # joint_name = getattr(joint, "name", full_identifier.split("/")[-1])
                    if len(full_identifier) == 0 or len(full_identifier.split("/")[-1]) == 0:
                        # logger.warning(f"Remove the attached dummy joint in object {obj_name}!")
                        continue
                    elif full_identifier not in self._buffer_dict[obj_name].joint_names:
                        self._buffer_dict[obj_name].joint_names.append(full_identifier)
                        if full_identifier in actuator_names:
                            self._buffer_dict[obj_name].actuator_names.append(full_identifier)
                        else:
                            logger.warning(f"Joint '{full_identifier}' in object '{obj_name}' is not actuated.")
                    else:
                        logger.error(f"Duplicate joint name detected: {full_identifier} in object {obj_name}")

            for body in all_bodies:
                # Check if body belongs to this object
                full_identifier = body.full_identifier
                if full_identifier and full_identifier.startswith(obj_prefix):
                    # Extract the body name without the prefix
                    # body_name = getattr(body, "name", full_identifier.split("/")[-1])
                    if len(full_identifier) == 0 or len(full_identifier.split("/")[-1]) == 0:
                        # logger.warning(f"Remove the attached dummy body in object {obj_name}!")
                        continue
                    elif full_identifier not in self._buffer_dict[obj_name].body_names:
                        self._buffer_dict[obj_name].body_names.append(full_identifier)
                    else:
                        logger.error(f"Duplicate body name detected: {full_identifier} in object {obj_name}")

            # Add sensors if configured for this object
            if hasattr(obj_cfg, "sensors") and obj_cfg.sensors:
                for sensor_name, sensor_cfg in obj_cfg.sensors.items():
                    sensor_type = sensor_cfg.type
                    if sensor_type in _SENSOR_TYPE_REGISTRY:
                        sensor_instance = _SENSOR_TYPE_REGISTRY[sensor_type](sensor_cfg)
                        # sensor_instance.bind(self, obj_name, sensor_name)
                        self._buffer_dict[obj_name].sensors[sensor_name] = sensor_instance
                    else:
                        logger.error(
                            f"Unsupported sensor type '{sensor_type}' for sensor '{sensor_name}' in object '{obj_name}'"
                        )
            self._buffer_dict[obj_name].config = obj_cfg

    def _set_root_state(self, obj_name: str, obj_state: ObjectState, env_ids: ArrayType):
        """Set root position and rotation."""

        identifier = obj_name + "/"  # MuJoCo joint/body names are prefixed with model name
        if not self._buffer_dict[obj_name].config.properties.get("fix_base_link", False):
            # when the object is not fixed-base-link
            root_joint = self._mjcf_physics.data.joint(identifier)
            root_joint.qpos[:7] = obj_state.root_state[env_ids, :7]
            root_joint.qvel[:6] = obj_state.root_state[env_ids, 7:13]
        # except KeyError:  # when the object is fixed-base-link, use the following method
        else:
            # body pos
            self._mjcf_physics.named.model.body_pos[identifier] = obj_state.root_state[env_ids, :3]
            # body quat
            self._mjcf_physics.named.model.body_quat[identifier] = obj_state.root_state[env_ids, 3:7]

    def _set_joint_state(self, obj_name: str, obj_state: ObjectState, env_ids: ArrayType):
        """Set joint positions."""
        joint_names = self.get_joint_names(obj_name)
        if len(joint_names) > 0:
            self._mjcf_physics.named.data.qpos[joint_names] = obj_state.joint_pos[env_ids]
            self._mjcf_physics.named.data.qvel[joint_names] = obj_state.joint_vel[env_ids]

        # for i, joint_name in enumerate(self.get_joint_names(obj_name)):
        #     # Here, we assume the data order is same with the order od the get_joint_names()
        #     joint = self._mjcf_physics.data.joint(joint_name)
        #     joint.qpos[:] = obj_state.joint_pos[env_ids, i]
        #     joint.qvel[:] = obj_state.joint_vel[env_ids, i]

    def _get_action_indices(self, name: str) -> np.ndarray:
        if self._buffer_dict[name].action_indices is None:
            actuator_names = self.get_actuator_names(name)
            action_indices = []
            for actuator_name in actuator_names:
                try:
                    actuator_id = self._mjcf_physics.model.actuator(actuator_name).id
                    action_indices.append(actuator_id)
                    # self._buffer_dict[name].actuator_names.append(actuator_name)
                except Exception:
                    logger.warning(f"Actuator '{actuator_name}' in object '{name}' cannot be found in MJCF Physics.")
            action_indices = np.array(action_indices, dtype=np.int32)
            self._buffer_dict[name].action_indices = action_indices
            return action_indices
        return self._buffer_dict[name].action_indices

    def _create_builtin_xml(self, obj_cfg: ObjectConfig) -> str:
        prop: dict[str, Any] = obj_cfg.properties
        if obj_cfg.type == ObjectType.CUBE:
            half_size = prop.get("half_size", [0.1, 0.1, 0.1])
            size_str = f"{half_size[0]} {half_size[1]} {half_size[2]}"
            type_str = "box"
        elif obj_cfg.type == ObjectType.CYLINDER:
            radius = prop.get("radius", 0.1)
            height = prop.get("height", 0.1)
            size_str = f"{radius} {height}"
            type_str = "cylinder"
        elif obj_cfg.type == ObjectType.SPHERE:
            radius = prop.get("radius", 0.1)
            size_str = f"{radius}"
            type_str = "sphere"
        else:
            raise ValueError(
                f"Unknown object type, available types: {[t.value for t in ObjectType if t not in [ObjectType.CUSTOM, ObjectType.ROBOT]]}."
            )
        color = prop.get("color", [0.5, 0.5, 0.5])
        rgba_str = f"{color[0]} {color[1]} {color[2]} 1"
        density = prop.get("density", 1000.0)
        xml = f"""
        <mujoco model="{type_str}">
        <worldbody>
            <body name="{type_str}_body" pos="{0} {0} {0}">
            <geom name="{type_str}_geom" type="{type_str}" size="{size_str}" rgba="{rgba_str}" density="{density}"/>
            </body>
        </worldbody>
        </mujoco>
        """

        return xml.strip()

    # def _add_cameras(self, mjcf_model: mjcf.RootElement) -> None:
    #     """Add cameras to the model."""
    #     camera_max_width = 640
    #     camera_max_height = 480

    #     for camera in self.cameras:
    #         direction = np.array(
    #             [
    #                 camera.look_at[0] - camera.pos[0],
    #                 camera.look_at[1] - camera.pos[1],
    #                 camera.look_at[2] - camera.pos[2],
    #             ]
    #         )
    #         direction = direction / np.linalg.norm(direction)
    #         up = np.array([0, 0, 1])
    #         right = np.cross(direction, up)
    #         right = right / np.linalg.norm(right)
    #         up = np.cross(right, direction)

    #         camera_params = {
    #             "pos": f"{camera.pos[0]} {camera.pos[1]} {camera.pos[2]}",
    #             "mode": "fixed",
    #             "fovy": camera.vertical_fov,
    #             "xyaxes": f"{right[0]} {right[1]} {right[2]} {up[0]} {up[1]} {up[2]}",
    #         }
    #         mjcf_model.worldbody.add("camera", name=f"{camera.name}_custom", **camera_params)
    #         camera_max_width = max(camera_max_width, camera.width)
    #         camera_max_height = max(camera_max_height, camera.height)

    #     if camera_max_width > 640 or camera_max_height > 480:
    #         self._set_framebuffer_size(mjcf_model, camera_max_width, camera_max_height)

    # def _get_actuator_states(self, obj_name):
    #     """Get actuator states (targets and forces)."""
    #     actuator_states = {
    #         "dof_pos_target": {},
    #         "dof_vel_target": {},
    #         "dof_torque": {},
    #     }

    #     # Find the robot index
    #     robot_idx = None
    #     for i, robot in enumerate(self.robots):
    #         if robot.name == obj_name:
    #             robot_idx = i
    #             break

    #     if robot_idx is None:
    #         return actuator_states

    #     robot_name = self._mujoco_robot_names[robot_idx]

    #     for actuator_id in range(self._mjcf_physics.model.nu):
    #         actuator = self._mjcf_physics.model.actuator(actuator_id)
    #         if actuator.name.startswith(robot_name):
    #             clean_name = actuator.name[len(robot_name) :]

    #             actuator_states["dof_pos_target"][clean_name] = float(
    #                 self._mjcf_physics.data.ctrl[actuator_id].item()
    #             )  # Hardcoded to position control
    #             actuator_states["dof_vel_target"][clean_name] = None
    #             actuator_states["dof_torque"][clean_name] = float(self._mjcf_physics.data.actuator_force[actuator_id].item())

    #     return actuator_states

    # def _pack_state(self, body_ids: list[int]):
    #     """
    #     Pack pos(3), quat(4), lin_vel_world(3), ang_vel(3) for one-env MuJoCo.

    #     Args:
    #         body_ids: list of body IDs, e.g. [root_id] or [root_id] + body_ids_reindex

    #     Returns:
    #         root_np: numpy (13,)      — the first body
    #         body_np: numpy (n_body,13)     — n_body bodies
    #     """
    #     data = self._mjcf_physics.data
    #     pos = data.xpos[body_ids]
    #     quat = data.xquat[body_ids]

    #     # angular ω (world) & v @ subtree_com
    #     w = data.cvel[body_ids, 0:3]
    #     v = data.cvel[body_ids, 3:6]

    #     # compute world‐frame linear velocity at body origin
    #     offset = data.xpos[body_ids] - data.subtree_com[body_ids]
    #     lin_world = v + np.cross(w, offset)

    #     full = np.concatenate([pos, quat, lin_world, w], axis=1)
    #     root_np = full[0]
    #     return root_np, full[1:]  # root, bodies

    # def _mirror_state_to_native(self):
    #     self._mj_data = self._mjcf_physics._data._data

    # def _set_root_state(self, obj_name, obj_state, zero_vel=False):
    #     """Set root position and rotation."""
    #     if "pos" not in obj_state and "rot" not in obj_state:
    #         return

    #     # Check if it's a robot
    #     robot_idx = None
    #     for i, robot in enumerate(self.robots):
    #         if robot.name == obj_name:
    #             robot_idx = i
    #             break

    #     if robot_idx is not None:
    #         robot = self.robots[robot_idx]
    #         robot_name = self._mujoco_robot_names[robot_idx]
    #         if not robot.fix_base_link:
    #             root_joint = self._mjcf_physics.data.joint(robot_name)
    #             root_joint.qpos[:3] = obj_state.get("pos", [0, 0, 0])
    #             root_joint.qpos[3:7] = obj_state.get("rot", [1, 0, 0, 0])
    #             if zero_vel:
    #                 root_joint.qvel[:6] = 0
    #         else:
    #             root_body = self._mjcf_physics.named.model.body_pos[robot_name]
    #             root_body_quat = self._mjcf_physics.named.model.body_quat[robot_name]
    #             root_body[:] = obj_state.get("pos", [0, 0, 0])
    #             root_body_quat[:] = obj_state.get("rot", [1, 0, 0, 0])
    #     else:
    #         model_name = self.mj_objects[obj_name].model + "/"
    #         try:
    #             obj_joint = self._mjcf_physics.data.joint(model_name)
    #             obj_joint.qpos[:3] = obj_state["pos"]
    #             obj_joint.qpos[3:7] = obj_state["rot"]
    #             if zero_vel:
    #                 obj_joint.qvel[:6] = 0
    #         except KeyError:
    #             obj_body = self._mjcf_physics.named.model.body_pos[model_name]
    #             obj_body_quat = self._mjcf_physics.named.model.body_quat[model_name]
    #             obj_body[:] = obj_state["pos"]
    #             obj_body_quat[:] = obj_state["rot"]

    # def _set_joint_state(self, obj_name, obj_state, zero_vel=False):
    #     """Set joint positions."""
    #     if "dof_pos" not in obj_state:
    #         return

    #     # Check if it's a robot
    #     robot_idx = None
    #     for i, robot in enumerate(self.robots):
    #         if robot.name == obj_name:
    #             robot_idx = i
    #             break

    #     for joint_name, joint_pos in obj_state["dof_pos"].items():
    #         if robot_idx is not None:
    #             robot_name = self._mujoco_robot_names[robot_idx]
    #             full_joint_name = f"{robot_name}{joint_name}"
    #         else:
    #             full_joint_name = f"{obj_name}/{joint_name}"

    #         joint = self._mjcf_physics.data.joint(full_joint_name)
    #         joint.qpos = joint_pos
    #         if zero_vel:
    #             joint.qvel = 0
    #         try:
    #             actuator = self._mjcf_physics.model.actuator(full_joint_name)
    #             self._mjcf_physics.data.ctrl[actuator.id] = joint_pos
    #         except KeyError:
    #             pass

    # def _disable_robotgravity(self):
    #     gravity_vec = np.array(self.scenario.gravity)

    #     self._mjcf_physics.data.xfrc_applied[:] = 0
    #     for body_name in self.robot_body_names:
    #         body_id = self._mjcf_physics.model.body(body_name).id
    #         force_vec = -gravity_vec * self._mjcf_physics.model.body(body_name).mass
    #         self._mjcf_physics.data.xfrc_applied[body_id, 0:3] = force_vec
    #         self._mjcf_physics.data.xfrc_applied[body_id, 3:6] = 0

    # def refresh_render(self) -> None:
    #     self._mjcf_physics.forward()  # Recomputes the forward dynamics without advancing the simulation.
    #     if self.viewer is not None:
    #         self.viewer.sync()

    # def _get_camera_params(self, camera_id: str, camera):
    #     """Get camera intrinsics and extrinsics from MuJoCo camera configuration.

    #     Returns:
    #         Ks: (3, 3) intrinsic matrix
    #         c2w: (4, 4) camera-to-world transformation matrix
    #     """
    #     mj_camera = self._mjcf_physics.model.camera(camera_id)

    #     # Extrinsics: build from camera configuration
    #     cam_pos = self._mjcf_physics.data.cam_xpos[mj_camera.id]

    #     # Compute camera orientation from pos and look_at
    #     forward = np.array(camera.look_at) - np.array(camera.pos)
    #     forward = forward / np.linalg.norm(forward)

    #     world_up = np.array([0, 0, 1])
    #     right = np.cross(forward, world_up)
    #     right = right / np.linalg.norm(right)
    #     up = np.cross(right, forward)

    #     # Build c2w matrix (OpenGL convention: camera looks along -Z)
    #     c2w = np.eye(4)
    #     c2w[:3, 0] = right
    #     c2w[:3, 1] = up
    #     c2w[:3, 2] = -forward  # Z axis points backward
    #     c2w[:3, 3] = cam_pos

    #     # Intrinsics: compute from vertical FOV
    #     fovy_rad = np.deg2rad(camera.vertical_fov)
    #     fy = camera.height / (2 * np.tan(fovy_rad / 2))
    #     fx = fy  # assume square pixels
    #     cx = camera.width / 2.0
    #     cy = camera.height / 2.0
    #     Ks = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    #     return Ks, c2w

    # def _apply_scale_to_mjcf(self, mjcf_model, scale):
    #     """Apply scale to all geoms, bodies, and sites in the MJCF model."""
    #     scale_x, scale_y, scale_z = scale

    #     for geom in mjcf_model.find_all("geom"):
    #         if hasattr(geom, "size") and geom.size is not None:
    #             size = list(geom.size)
    #             if geom.type in ["box", None]:
    #                 if len(size) >= 3:
    #                     geom.size = [size[0] * scale_x, size[1] * scale_y, size[2] * scale_z]
    #             elif geom.type == "sphere":
    #                 if len(size) >= 1:
    #                     geom.size = [size[0] * max(scale_x, scale_y, scale_z)]
    #             elif geom.type == "cylinder":
    #                 if len(size) >= 2:
    #                     radius_scale = max(scale_x, scale_y)
    #                     geom.size = [size[0] * radius_scale, size[1] * scale_z]
    #             elif geom.type == "capsule":
    #                 if len(size) >= 2:
    #                     radius_scale = max(scale_x, scale_y)
    #                     geom.size = [size[0] * radius_scale, size[1] * scale_z]
    #             elif geom.type == "ellipsoid":
    #                 if len(size) >= 3:
    #                     geom.size = [size[0] * scale_x, size[1] * scale_y, size[2] * scale_z]

    #         if hasattr(geom, "pos") and geom.pos is not None:
    #             pos = list(geom.pos)
    #             if len(pos) >= 3:
    #                 geom.pos = [pos[0] * scale_x, pos[1] * scale_y, pos[2] * scale_z]

    #     for body in mjcf_model.find_all("body"):
    #         if hasattr(body, "pos") and body.pos is not None:
    #             pos = list(body.pos)
    #             if len(pos) >= 3:
    #                 body.pos = [pos[0] * scale_x, pos[1] * scale_y, pos[2] * scale_z]

    #     for site in mjcf_model.find_all("site"):
    #         if hasattr(site, "pos") and site.pos is not None:
    #             pos = list(site.pos)
    #             if len(pos) >= 3:
    #                 site.pos = [pos[0] * scale_x, pos[1] * scale_y, pos[2] * scale_z]

    #         if hasattr(site, "size") and site.size is not None:
    #             size = list(site.size)
    #             if len(size) >= 1:
    #                 site.size = [size[0] * max(scale_x, scale_y, scale_z)]

    #     for joint in mjcf_model.find_all("joint"):
    #         if hasattr(joint, "pos") and joint.pos is not None:
    #             pos = list(joint.pos)
    #             if len(pos) >= 3:
    #                 joint.pos = [pos[0] * scale_x, pos[1] * scale_y, pos[2] * scale_z]

    #     # Apply scale to mesh elements (for visual meshes)
    #     for mesh in mjcf_model.find_all("mesh"):
    #         if hasattr(mesh, "scale") and mesh.scale is not None:
    #             mesh_scale = list(mesh.scale)
    #             if len(mesh_scale) >= 3:
    #                 mesh.scale = [
    #                     mesh_scale[0] * scale_x,
    #                     mesh_scale[1] * scale_y,
    #                     mesh_scale[2] * scale_z,
    #                 ]
    #             elif len(mesh_scale) == 1:
    #                 # Uniform scale
    #                 uniform_scale = max(scale_x, scale_y, scale_z)
    #                 mesh.scale = [mesh_scale[0] * uniform_scale]

    # def _set_framebuffer_size(self, mjcf_model, width, height):
    #     visual_elem = mjcf_model.visual
    #     global_elem = None
    #     for child in visual_elem._children:
    #         if child.tag == "global":
    #             global_elem = child
    #             break
    #     if global_elem is None:
    #         global_elem = visual_elem.add("global")
    #     global_elem.offwidth = width
    #     global_elem.offheight = height

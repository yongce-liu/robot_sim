import os
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Callable

import cv2
import mujoco
import mujoco.viewer
import numpy as np
from dm_control import mjcf
from loguru import logger

from robot_sim.backends.base import BaseBackend
from robot_sim.configs import ObjectConfig, ObjectType, TerrainType
from robot_sim.configs.types import ActionsType, ArrayType, ObjectState, StatesType
from robot_sim.utils.helper import recursive_setattr, resolve_asset_path


class MujocoBackend(BaseBackend):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._actions_cache: ActionsType = {}  # robot: action
        assert self.num_envs == 1, f"Mujoco only supports single env, got {self.num_envs}."
        assert self.device == "cpu", f"Mujoco only supports CPU device, got {self.device}."

        # self._mjcf_sub_models: dict[str, mjcf.RootElement] = {}  # robot/object name -> mjcf model
        self._mjcf_model: mjcf.RootElement = None
        self._mjcf_physics: mjcf.Physics = None
        self._renderer: Callable | None = None
        self._mjcf_model = self._init_mujoco()
        self._update_object_model()

    def _init_mujoco(self) -> mjcf.RootElement:
        if self._config.scene.path is not None:
            _asset_path = resolve_asset_path(self._config.scene.path)
            mjcf_model = mjcf.from_path(_asset_path)
            logger.info(f"Loaded scene from: {_asset_path}")
        else:
            mjcf_model = mjcf.RootElement()

        self._add_terrain(mjcf_model)
        self._add_visual(mjcf_model)
        self._add_objects(mjcf_model)
        # dt
        mjcf_model.option.timestep = self.cfg_sim.dt
        # gravity
        mjcf_model.option.gravity = self.cfg_sim.gravity
        self.export_mjcf(model=mjcf_model, out_dir="outputs/mujoco/", file_name="robot_sim.xml")

        return mjcf_model

    @staticmethod
    def __update_from_set(obj_name: str, sets: list[mjcf.Element]) -> list[str]:
        # In dm_control, attached elements are prefixed with the model name
        obj_prefix = f"{obj_name}/"
        ans = []
        for sub in sets:
            # Check if joint belongs to this object by checking the full_identifier
            full_identifier = sub.full_identifier
            if full_identifier and full_identifier.startswith(obj_prefix):
                # Extract the joint name without the prefix
                name = full_identifier[len(obj_prefix) :]
                if len(full_identifier) == 0 or len(name) == 0:
                    # logger.warning(f"Remove the attached dummy joint in object {obj_name}!")
                    continue
                elif name not in ans:
                    ans.append(name)
                else:
                    logger.error(f"Duplicate name detected: {name} in object {obj_name}")
        return ans

    def _update_object_model(self) -> None:
        """Update joint and body name indices for the given model."""
        model: mjcf.RootElement = self._mjcf_model
        # all_joints = model.find_all("joint")
        all_bodies = model.find_all("body")
        # all_actuators = model.find_all("actuator")
        for name, obj in self.objects.items():
            if obj.body_names is None:
                obj.body_names = self.__update_from_set(name, all_bodies)

    def _launch(self) -> None:
        """Initialize MuJoCo model with optional scene support."""

        assert self._mjcf_model is not None, "MuJoCo model must be initialized."
        self._mjcf_physics = mjcf.Physics.from_mjcf_model(self._mjcf_model)

        # Export MJCF + assets to a temp dir.
        # Handle filename variability (dm_control 1.0.34).

        # FIXME: whether need to reload the model?
        # # Load the model from the XML *in the same directory as the exported assets*
        # # so hashed filenames resolve correctly.
        # self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
        # self._mj_data = mujoco.MjData(self._mj_model)
        if not self.headless:
            self._init_renderer()

    def _init_renderer(self) -> None:
        self._render_cfg: dict = self.cfg_spec.get("render", {})
        self._render_cfg["mode"] = self._render_cfg.get("mode", "mjviewer")
        if self._render_cfg["mode"] == "mjviewer":
            self.__viewer = mujoco.viewer.launch_passive(
                self._mjcf_physics.model.ptr, self._mjcf_physics.data.ptr, show_left_ui=False, show_right_ui=False
            )
            self.__viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = 0
            self.__viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
            self.__viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0
            self.__viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0
            self._renderer = self.__viewer.sync
        elif self._render_cfg["mode"] == "opencv":
            self._renderer = partial(
                self._mjcf_physics.render,
                camera_id=self._render_cfg.get("camera", 0),
                width=self._render_cfg.get("width", 640),
                height=self._render_cfg.get("height", 480),
            )
        else:
            raise ValueError(f"Unknown render mode: {self._render_cfg['mode']}")

    def _render(self) -> None:
        if self._renderer is not None:
            if self._render_cfg["mode"] == "opencv":
                rgb_array = self._renderer().copy()
                cv2.imshow("Mujoco Render", cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR))
                # Ensure the OpenCV window processes events and refreshes.
                cv2.waitKey(1)
            elif self._render_cfg["mode"] == "mjviewer":
                self._renderer()

    def get_rgb_image(self) -> np.ndarray | Any | None:
        """Get the RGB image of the environment.

        Returns:
            np.ndarray: The RGB image of the environment.
        """
        try:
            assert self._mjcf_physics is not None, "MuJoCo physics must be initialized."
            return self._mjcf_physics.render(
                camera_id=self._render_cfg.get("camera", 0),
                width=self._render_cfg.get("width", 640),
                height=self._render_cfg.get("height", 480),
            ).copy()
        except Exception:
            logger.warning(
                "There are no cameras available in the MuJoCo environment. You can add cameras for the simulator, and choose the camera id to render images."
            )
            return None

    def close(self):
        if self._renderer is not None:
            if self._render_cfg["mode"] == "mjrender":
                cv2.destroyAllWindows()
            if hasattr(self, "__viewer"):
                self.__viewer.close()

    def _simulate(self):
        # # Apply gravity compensation for all robots
        # for robot_name, robot_cfg in enumerate(self.robots):
        #     if self._gravity_compensations[robot_name]:
        #         self._disable_robotgravity()

        self._mjcf_physics.step()

    def _set_states(self, states: StatesType, env_ids: ArrayType):
        """Set the states of all objects and robots."""

        for obj_name, obj_state in states.items():
            self._set_root_state(obj_name, obj_state, env_ids)
            self._set_joint_state(obj_name, obj_state, env_ids)
        self._mjcf_physics.forward()

    def _set_actions(self, actions: ActionsType, env_ids: ArrayType) -> None:
        """Set actions for all robots/objects with ctrl entrypoint."""
        self._actions_cache = actions  # dict[str, np.ndarray (num_envs, num_dofs)]
        for obj_name, obj_action in actions.items():
            actuator_names = self.objects[obj_name].get_actuator_names(prefix=f"{obj_name}/")
            # Here, we also use the default order index when adding a object/robot
            self._mjcf_physics.named.data.ctrl[actuator_names] = obj_action[env_ids]

    def _get_states(self, dtype=np.float32) -> StatesType:
        """Get states of all objects and robots."""

        states: dict[str, ObjectState] = {}
        _pnd = self._mjcf_physics.named.data
        for obj_name in self.cfg_objects.keys():
            joint_names = self.objects[obj_name].get_joint_names(prefix=f"{obj_name}/")
            _root_state, _body_state = self._pack_state(obj_name)
            state = ObjectState(
                root_state=_root_state.astype(dtype).copy(),
                body_state=_body_state.astype(dtype).copy(),
                joint_pos=_pnd.qpos[joint_names].copy()[None, ...].astype(dtype) if joint_names else None,
                joint_vel=_pnd.qvel[joint_names].copy()[None, ...].astype(dtype) if joint_names else None,
                joint_action=None,
                sensors={k: deepcopy(v.data) for k, v in self._sensors[obj_name].items()},
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
        body_names = [f"{obj_name}/"] + self.objects[obj_name].get_body_names(prefix=f"{obj_name}/")
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
        if self.cfg_terrain is None:
            logger.warning("No terrain configuration provided; skipping terrain addition.")
            return
        texture_name = "texplane"
        material_name = "matplane"
        geom_name = "ground_plane"
        if self.cfg_terrain.type == TerrainType.CUSTOM:
            terrain_mjcf = mjcf.from_path(self.cfg_terrain.path)
            terrain_mjcf.model = "terrain"
            terrain_mjcf.find("texture")[0].name = texture_name
            terrain_mjcf.find("material")[0].name = material_name
            terrain_mjcf.find("geom")[0].name = geom_name
            model.worldbody.add(terrain_mjcf)
            logger.info(f"Loaded terrain from: {self.cfg_terrain.path}")
        elif self.cfg_terrain.type == TerrainType.PLANE:
            model.asset.add(
                "texture",
                name=texture_name,
                type="2d",
                builtin="checker",
                width=512,
                height=512,
                rgb1=[0.2, 0.3, 0.4],
                rgb2=[0.1, 0.2, 0.3],
            )
            model.asset.add("material", name=material_name, texture=texture_name, texrepeat=[2, 2], texuniform=True)
            model.worldbody.add("geom", name=geom_name, type="plane", size="0 0 0.001", material=material_name)
        else:
            raise ValueError(f"Unknown terrain type: {self.cfg_terrain.type}")

        recursive_setattr(model.asset.texture[texture_name], self.cfg_terrain.properties.get("texture"), tag="texture")
        recursive_setattr(
            model.asset.material[material_name], self.cfg_terrain.properties.get("material"), tag="material"
        )
        recursive_setattr(model.worldbody.geom[geom_name], self.cfg_terrain.properties.get("geom"), tag="geom")

    def _add_visual(self, model: mjcf.RootElement) -> None:
        if self.cfg_visual is None:
            logger.warning("No visual configuration provided; skipping visual addition.")
            return

        recursive_setattr(model.visual.headlight, self.cfg_visual.light, tag="headlight")
        recursive_setattr(model.visual, self.cfg_visual.properties, tag="visual")

    def _add_objects(self, model: mjcf.RootElement) -> None:
        """Add individual objects to the model."""
        for obj_name, obj_cfg in self.cfg_objects.items():
            if obj_cfg.type in [ObjectType.CUSTOM, ObjectType.ROBOT]:
                _asset_path = resolve_asset_path(obj_cfg.path)
                obj_mjcf = mjcf.from_path(_asset_path)
                logger.info(f"Loaded object '{obj_name}' from: {_asset_path}")
                # TODO: handle free joints for custom objects
                """
                # Remove free joint since dm_control has limit support for it.
                for joint in obj_mjcf.find_all("joint"):
                    if joint.tag == "joint" and joint.type == "free":
                        joint.remove()
                """
                # add armature
                for joint in obj_mjcf.find_all("joint"):
                    if joint.tag != "joint":
                        continue
                    if getattr(joint, "type", None) == "free":
                        continue
                    assert obj_cfg.joints is not None, f"Object {obj_name} has joints, but no joint config provided."
                    for prop_key, prop_val in obj_cfg.joints[joint.name].properties.items():
                        try:
                            setattr(joint, prop_key, prop_val)
                        except AttributeError as e:
                            logger.warning(f"Joint {joint.name} does not support property '{prop_key}': {e}")
            else:
                xml_str = self._create_builtin_xml(obj_cfg)
                obj_mjcf = mjcf.from_xml_string(xml_str)

            obj_mjcf.model = obj_name
            obj_attached = model.attach(obj_mjcf)
            obj_attached.pos = obj_cfg.pose[0:3]
            obj_attached.quat = obj_cfg.pose[3:7]

            # FIXME: temporary fix for free joint handling in dm_control
            self._fix_attach_bugs(obj_attached, obj_name, obj_cfg)
            # self._mjcf_sub_models[obj_name] = obj_mjcf

    @staticmethod
    def _fix_attach_bugs(obj_attached: mjcf.RootElement, obj_name: str, obj_cfg: ObjectConfig) -> None:
        """
        # FIXME: A temporary workaround for free joint handling in dm_control
        temporaly fix attaching robots bugs in dm_control when attach a object.
        """

        all_joints = obj_attached.find_all("joint")
        if len(all_joints) > 0:
            child_joint = all_joints[0]
            if child_joint.tag == "freejoint" or getattr(child_joint, "type", None) == "free":
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
    def export_mjcf(model: mjcf.RootElement, out_dir: str | os.PathLike, file_name: str = "robot_sim.xml") -> None:
        """Export the full MJCF model and assets to the specified directory."""
        # Write assets + XML to disk (this version returns None and writes files)
        # model_name is not guaranteed to be respected by all versions, so we’ll glob later.
        model.model = file_name.split(".")[0]
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        mjcf.export_with_assets(model, out_dir, out_file_name=file_name)
        logger.info(f"Exported MJCF model and assets to: {out_dir}/{file_name}")

    def _set_root_state(self, obj_name: str, obj_state: ObjectState, env_ids: ArrayType):
        """Set root position and rotation."""

        identifier = obj_name + "/"  # MuJoCo joint/body names are prefixed with model name
        if not self.cfg_objects[obj_name].properties.get("fix_base_link", False):
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
        joint_names = self.objects[obj_name].get_joint_names(prefix=f"{obj_name}/")
        if joint_names is not None and obj_state.joint_pos is not None and obj_state.joint_vel is not None:
            self._mjcf_physics.named.data.qpos[joint_names] = obj_state.joint_pos[env_ids]
            self._mjcf_physics.named.data.qvel[joint_names] = obj_state.joint_vel[env_ids]

        # for i, joint_name in enumerate(self.get_joint_names(obj_name)):
        #     # Here, we assume the data order is same with the order od the get_joint_names()
        #     joint = self._mjcf_physics.data.joint(joint_name)
        #     joint.qpos[:] = obj_state.joint_pos[env_ids, i]
        #     joint.qvel[:] = obj_state.joint_vel[env_ids, i]

    def _create_builtin_xml(self, obj_cfg: ObjectConfig) -> str:
        prop: dict[str, Any] = obj_cfg.properties
        if obj_cfg.type == ObjectType.CUBE:
            size = prop.get("size", [0.1, 0.1, 0.1])
            size_str = f"{size[0] / 2} {size[1] / 2} {size[2] / 2}"
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

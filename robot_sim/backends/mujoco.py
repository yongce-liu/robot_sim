from dataclasses import asdict

import mujoco
import numpy as np
import torch
from dm_control import mjcf
from loguru import logger as log

try:
    # Patch mujoco.renderer.Renderer.__del__ to avoid noisy TypeError if glfw.free is None
    if hasattr(mujoco, "renderer") and hasattr(mujoco.renderer, "Renderer"):
        _orig_renderer_del = getattr(mujoco.renderer.Renderer, "__del__", None)

        def _safe_renderer_del(self):
            try:
                if _orig_renderer_del is not None:
                    _orig_renderer_del(self)
            except Exception:
                # Swallow exceptions during interpreter shutdown
                return

        mujoco.renderer.Renderer.__del__ = _safe_renderer_del

    # Patch mujoco.glfw.GLContext.__del__ similarly if present
    if hasattr(mujoco, "glfw") and hasattr(mujoco.glfw, "GLContext"):
        _orig_glctx_del = getattr(mujoco.glfw.GLContext, "__del__", None)

        def _safe_glctx_del(self):
            try:
                if _orig_glctx_del is not None:
                    _orig_glctx_del(self)
            except Exception:
                return

        mujoco.glfw.GLContext.__del__ = _safe_glctx_del
except Exception:
    pass

import os
import sys
import tempfile
from typing import Any

import mujoco.viewer

from robot_sim.backends.base import ActionType, BaseBackend
from robot_sim.configs import ObjectType, SimulatorConfig


class MujocoBackend(BaseBackend):
    def __init__(self, config: SimulatorConfig, optional_queries: dict[str, Any] | None = None):
        super().__init__(config, optional_queries)
        self._actions_cache: ActionType = {}  # robot: action
        assert self.num_envs == 1, f"Mujoco only supports single env, got {self.num_envs}."
        assert self.device == "cpu", f"Mujoco only supports CPU device, got {self.device}."

        self._mjcf_sub_models: dict[str, mjcf.RootElement] = {}  # robot/object name -> mjcf model
        self._mjcf_model: mjcf.RootElement | None = None
        self._mjcf_physics: mjcf.Physics | None = None
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

    def _launch(self) -> None:
        """Initialize MuJoCo model with optional scene support."""
        if self.config.scene.path is not None:
            mjcf_model = mjcf.from_path(self.config.scene.path)
            log.info(f"Loaded scene from: {self.config.scene.path}")
        else:
            mjcf_model = mjcf.RootElement()
            self._add_terrain(mjcf_model)
        self._update_indices(mjcf_model)  # update the root model indices first

        # self._add_cameras(model)
        self._add_objects(mjcf_model)
        self._add_robots(mjcf_model)
        for k, v in asdict(self.cfg_phyx).items():
            if hasattr(mjcf_model.option, k):
                setattr(mjcf_model.option, k, v)  # dt, gravity, etc.

        self._mjcf_model = mjcf_model
        self._mjcf_physics = mjcf.Physics.from_mjcf_model(mjcf_model)

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

    def render(self) -> None:
        pass

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
        if self.renderer is not None:
            try:
                self.renderer.close()
            except Exception:
                pass
            self.renderer = None

    def _simulate(self):
        # Apply gravity compensation for all robots
        for robot_idx, robot in enumerate(self.robots):
            if self._gravity_compensations[robot_idx]:
                self._disable_robotgravity()

        # Apply torque control if manual PD is enabled
        self.physics.step(self.decimation)

        if not self.headless:
            self.viewer.sync()

    def _set_states(self, states, env_ids=None, zero_vel=True):
        if isinstance(states, TensorState):
            states = state_tensor_to_nested(self, states)
        if len(states) > 1:
            raise ValueError("MujocoHandler only supports single env state setting")

        states_flat = [{**state["objects"], **state["robots"]} for state in states]

        for obj_name, obj_state in states_flat[0].items():
            if obj_name in self.mj_objects:
                self._set_root_state(obj_name, obj_state, zero_vel)
                self._set_joint_state(obj_name, obj_state, zero_vel)
        self.physics.forward()

    def _set_actions(self, actions) -> None:
        """Unified: Tensor/ndarray -> write ctrl (or cache for PD); dict-list -> name-based."""
        self._actions_cache = actions

        # Fast path: tensor-like controls
        if isinstance(actions, torch.Tensor):
            actions = actions.squeeze()
            vec = actions.detach().to(dtype=torch.float32, device="cpu").numpy()
            robot_idx = 0
            joint_names = self.get_joint_names(self.robot.name, sort=True)
            for i in range(self._robot_num_dofs[robot_idx]):
                joint_name = joint_names[i]
                actuator_id = self.physics.model.actuator(f"{self._mujoco_robot_names[robot_idx]}{joint_name}").id
                self.physics.data.ctrl[actuator_id] = vec[i]
            return

        # Dict-list path
        if isinstance(actions, list):  # Handling single-env parallell case
            actions = actions[0]
        for robot_idx, robot in enumerate(self.robots):
            payload = actions[robot.name]

            # Optional velocity targets
            vel_targets = payload.get("dof_vel_target")
            if vel_targets:
                jnames = self._get_joint_names(robot.name, sort=True)
                self._current_vel_target = np.zeros(self._robot_num_dofs[robot_idx], dtype=np.float32)
                for i, jn in enumerate(jnames):
                    if jn in vel_targets:
                        self._current_vel_target[i] = vel_targets[jn]
            else:
                self._current_vel_target = None

            # Position targets
            joint_targets = payload["dof_pos_target"]
            jnames = self._get_joint_names(robot.name, sort=True)
            self._current_action = np.zeros(self._robot_num_dofs[robot_idx], dtype=np.float32)
            for i, jn in enumerate(jnames):
                if jn in joint_targets:
                    self._current_action[i] = joint_targets[jn]
            for i in range(self._robot_num_dofs[robot_idx]):
                jn = jnames[i]
                if jn in joint_targets:
                    self.physics.data.actuator(f"{self._mujoco_robot_names[robot_idx]}{jn}").ctrl = joint_targets[jn]

    def _get_states(self, env_ids: list[int] | None = None) -> list[dict]:
        """Get states of all objects and robots."""
        object_states = {}

        # print("=== MuJoCo body names & positions ===")
        # for i in range(self.physics.model.nbody):
        #     body_name = self.physics.model.body(i).name
        #     body_pos = self.physics.data.xpos[i]  # (3,) np.array
        #     print(f"[{i}] {body_name} : pos = {body_pos}")
        # print("=====================================")
        for obj in self.objects:
            model_name = self.mj_objects[obj.name].model

            obj_body_id = self.physics.model.body(f"{model_name}/").id
            if isinstance(obj, ArticulationObjCfg):
                joint_names = self._get_joint_names(obj.name, sort=True)
                body_ids_reindex = self._get_body_ids_reindex(obj.name)

                root_np, body_np = self._pack_state([obj_body_id] + body_ids_reindex)
                state = ObjectState(
                    root_state=torch.from_numpy(root_np).float().unsqueeze(0),  # (1,13)
                    body_names=self._get_body_names(obj.name),
                    body_state=torch.from_numpy(body_np).float().unsqueeze(0),  # (1,n_body,13)
                    joint_pos=torch.tensor(
                        [self.physics.data.joint(f"{model_name}/{jn}").qpos.item() for jn in joint_names]
                    ).unsqueeze(0),
                    joint_vel=torch.tensor(
                        [self.physics.data.joint(f"{model_name}/{jn}").qvel.item() for jn in joint_names]
                    ).unsqueeze(0),
                )
            else:
                root_np, _ = self._pack_state([obj_body_id])

                state = ObjectState(
                    root_state=torch.from_numpy(root_np).float().unsqueeze(0),  # (1,13)
                )
            object_states[obj.name] = state

        robot_states = {}
        for robot in self.robots:
            model_name = self.mj_objects[robot.name].model
            obj_body_id = self.physics.model.body(f"{model_name}/").id
            joint_names = self._get_joint_names(robot.name, sort=True)
            actuator_reindex = self._get_actuator_reindex(robot.name)
            body_ids_reindex = self._get_body_ids_reindex(robot.name)
            root_np, body_np = self._pack_state([obj_body_id] + body_ids_reindex)

            state = RobotState(
                body_names=self._get_body_names(robot.name),
                root_state=torch.from_numpy(root_np).float().unsqueeze(0),  # (1,13)
                body_state=torch.from_numpy(body_np).float().unsqueeze(0),  # (1,n_body,13)
                joint_pos=torch.tensor(
                    [self.physics.data.joint(f"{model_name}/{jn}").qpos.item() for jn in joint_names]
                ).unsqueeze(0),
                joint_vel=torch.tensor(
                    [self.physics.data.joint(f"{model_name}/{jn}").qvel.item() for jn in joint_names]
                ).unsqueeze(0),
                joint_pos_target=torch.from_numpy(self.physics.data.ctrl[actuator_reindex]).unsqueeze(0),
                joint_vel_target=torch.from_numpy(self._current_vel_target).unsqueeze(0)
                if self._current_vel_target is not None
                else None,
                joint_effort_target=torch.from_numpy(self.physics.data.actuator_force[actuator_reindex]).unsqueeze(0),
            )
            robot_states[robot.name] = state

        camera_states = {}
        for camera in self.cameras:
            camera_id = f"{camera.name}_custom"  # XXX: hard code camera id for now

            depth = None

            if self.scenario.gs_scene is not None and self.scenario.gs_scene.with_gs_background:
                from metasim.utils.gs_util import alpha_blend_rgba

                # Extract camera parameters
                Ks, c2w = self._get_camera_params(camera_id, camera)

                # Render GS background
                gs_cam = SplatCamera.init_from_pose_list(
                    pose_list=c2w,
                    camera_intrinsic=Ks,
                    image_height=camera.height,
                    image_width=camera.width,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
                gs_result = self.gs_background.render(gs_cam)
                gs_result.to_numpy()

                # Get semantic segmentation (geom IDs and object IDs)
                sim_seg = self.physics.render(
                    width=camera.width, height=camera.height, camera_id=camera_id, depth=False, segmentation=True
                )
                geom_ids = sim_seg[..., 0] if sim_seg.ndim == 3 else sim_seg
                # Create foreground mask: exclude background (-1) and ground plane (0)
                foreground_mask = geom_ids >= 1
                seg_mask = np.where(foreground_mask, 255, 0).astype(np.uint8)

                if "rgb" in camera.data_types:
                    # Get MuJoCo simulation rendering
                    sim_rgb = self.physics.render(
                        width=camera.width, height=camera.height, camera_id=camera_id, depth=False, segmentation=False
                    )
                    # Blend RGB: foreground objects over GS background
                    sim_color = (sim_rgb * 255).astype(np.uint8) if sim_rgb.max() <= 1.0 else sim_rgb.astype(np.uint8)
                    foreground = np.concatenate([sim_color, seg_mask[..., None]], axis=-1)
                    background = gs_result.rgb.squeeze(0)
                    blended_rgb = alpha_blend_rgba(foreground, background)
                    rgb = torch.from_numpy(np.array(blended_rgb.copy()))

                if "depth" in camera.data_types:
                    sim_depth = self.physics.render(
                        width=camera.width, height=camera.height, camera_id=camera_id, depth=True, segmentation=False
                    )
                    # Compose depth: use simulation depth for foreground, GS depth for background
                    bg_depth = gs_result.depth.squeeze(0)
                    if bg_depth.ndim == 3 and bg_depth.shape[-1] == 1:
                        bg_depth = bg_depth[..., 0]
                    depth_comp = np.where(foreground_mask, sim_depth, bg_depth)
                    depth = torch.from_numpy(depth_comp.copy())

            else:
                if "rgb" in camera.data_types:
                    if sys.platform == "darwin":
                        with self._mj_lock:  # optional but safer
                            # match renderer size to camera if needed
                            if self.renderer is None or (self.renderer.width, self.renderer.height) != (
                                camera.width,
                                camera.height,
                            ):
                                self.renderer = mujoco.Renderer(
                                    self._mj_model, width=camera.width, height=camera.height
                                )
                            # mirror state and render
                            self._mirror_state_to_native()
                            self.renderer.update_scene(self._mj_data, camera=camera_id)
                            rgb_np = self.renderer.render()
                        rgb = torch.from_numpy(rgb_np.copy()).unsqueeze(0)
                    elif sys.platform == "win32":
                        rgb_np = self.physics.render(
                            width=camera.width, height=camera.height, camera_id=camera_id, depth=False
                        )
                        # Ensure numpy array -> torch tensor with shape (1, H, W, C)
                        rgb = torch.from_numpy(np.ascontiguousarray(rgb_np)).unsqueeze(0)
                    else:
                        rgb_np = self.physics.render(
                            width=camera.width, height=camera.height, camera_id=camera_id, depth=False
                        )
                        rgb = torch.from_numpy(np.ascontiguousarray(rgb_np)).unsqueeze(0)
                if "depth" in camera.data_types:
                    if sys.platform == "darwin":
                        with self._mj_lock:
                            # Ensure renderer matches the camera size
                            if self.renderer is None or (self.renderer.width, self.renderer.height) != (
                                camera.width,
                                camera.height,
                            ):
                                self.renderer = mujoco.Renderer(
                                    self._mj_model, width=camera.width, height=camera.height
                                )

                            # Keep native model/data in sync with dm_control physics
                            self._mirror_state_to_native()
                            self.renderer.update_scene(self._mj_data, camera=camera_id)

                            # --- Cross-version depth rendering for mujoco.Renderer ---
                            if hasattr(self.renderer, "enable_depth_rendering"):
                                # Newer MuJoCo (>= 3.2/3.3): enable depth mode, render(), then disable.
                                self.renderer.enable_depth_rendering()
                                depth_np = self.renderer.render()
                                self.renderer.disable_depth_rendering()
                            elif hasattr(mujoco, "RenderMode"):
                                # Some 3.x builds expose RenderMode enum on mujoco
                                depth_np = self.renderer.render(render_mode=mujoco.RenderMode.DEPTH)
                            else:
                                # Very old fallback: some builds returned (rgb, depth) as a tuple.
                                # If this still fails in your env, we’ll need a dedicated mjr_readPixels path.
                                maybe = self.renderer.render()
                                if isinstance(maybe, tuple) and len(maybe) == 2:
                                    _, depth_np = maybe
                                else:
                                    raise RuntimeError("Depth rendering not supported by this mujoco.Renderer build.")
                        depth = torch.from_numpy(depth_np.copy()).unsqueeze(0)
                    elif sys.platform == "win32":
                        depth_np = self.physics.render(
                            width=camera.width, height=camera.height, camera_id=camera_id, depth=True
                        )
                        depth = torch.from_numpy(np.ascontiguousarray(depth_np)).unsqueeze(0)
                    else:
                        depth_np = self.physics.render(
                            width=camera.width, height=camera.height, camera_id=camera_id, depth=True
                        )
                        depth = torch.from_numpy(np.ascontiguousarray(depth_np)).unsqueeze(0)
                state = CameraState(rgb=locals().get("rgb", None), depth=locals().get("depth", None))

            camera_states[camera.name] = state
        extras = self.get_extra()
        return TensorState(objects=object_states, robots=robot_states, cameras=camera_states, extras=extras)

    # Private methods for initializing MuJoCo model
    def _add_terrain(self, model: mjcf.RootElement) -> None:
        """Add default ground plane."""
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
        else:
            raise NotImplementedError(f"Terrain type '{self.terrain.type}' not supported yet in MuJoCo backend.")

    def _add_objects(self, model: mjcf.RootElement) -> None:
        """Add individual objects to the model."""
        for obj_name, obj_cfg in self.objects.items():
            if obj_cfg.type == ObjectType.CUSTOM:
                obj_mjcf = mjcf.from_path(obj_cfg.mjcf_path)
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
            # Apply default position and orientation to the object's root body,
            obj_attached.pos = obj_cfg.initial_position
            obj_attached.quat = obj_cfg.initial_orientation  # [w,x,y,z] format

            if not obj_cfg.properties.get("fix_base_link", False):  # default: False, assume the object can move freely
                obj_attached.add("freejoint")

            self._mjcf_sub_models[obj_name] = obj_mjcf
            self._update_indices(obj_mjcf)

    def _add_robots(self, model: mjcf.RootElement) -> None:
        """Add robots to the model."""
        for robot_name, robot_cfg in self.robots.items():
            robot_mjcf = mjcf.from_path(robot_cfg.path)
            robot_mjcf.model = robot_name
            robot_attached = model.attach(robot_mjcf)

            # FIXME: A temporary workaround for free joint handling in dm_control
            if not robot_cfg.properties.get("fix_base_link", False):  # default: False, assume the robot can move freely
                child_joint = robot_attached.find_all("joint")[0]
                if child_joint.type == "free":
                    log.warning(
                        f"Robot '{robot_name}' already has a free joint in its MJCF. "
                        "We will remove it and add the free joint at the root body level."
                    )
                    child_joint.remove()

                robot_attached.add("freejoint")

            # FIXME: Ensure the attached robot has an inertial element to avoid simulation issues
            if not hasattr(robot_attached, "inertial") or robot_attached.inertial is None:
                child_body = robot_attached.find_all("body")[0]
                pos = child_body.inertial.pos
                robot_attached.pos = child_body.pos
                child_body.pos = "0 0 0"  # Reset child body position to origin with respect to the attached robot
                robot_attached.quat = child_body.quat if child_body.quat is not None else "1 0 0 0"
                robot_attached.add("inertial", mass="1e-9", diaginertia="1e-9 1e-9 1e-9", pos=pos)

            self._mjcf_sub_models[robot_name] = robot_mjcf
            self._update_indices(robot_mjcf)

    @staticmethod
    def export_mjcf(model: mjcf.RootElement, out_dir: os.PathLike, file_name: str = "model.xml") -> None:
        """Export the full MJCF model and assets to the specified directory."""
        # Write assets + XML to disk (this version returns None and writes files)
        # model_name is not guaranteed to be respected by all versions, so we’ll glob later.
        mjcf.export_with_assets(model, out_dir, out_file_name=file_name)
        log.info(f"Exported MJCF model and assets to: {out_dir}")

    def _update_indices(self, model: mjcf.RootElement) -> None:
        """Update joint and body name indices for the given model."""
        obj_name = model.model
        for joint in model.find_all("joint"):
            if joint.name not in self._joint_indices[obj_name]:
                self._joint_indices[obj_name].append(joint.name)
            else:
                log.error(f"Duplicate joint name detected: {joint.name} in object {obj_name}")
        for body in model.find_all("body"):
            if body.name not in self._body_indices[obj_name]:
                self._body_indices[obj_name].append(body.name)
            else:
                log.error(f"Duplicate body name detected: {body.name} in object {obj_name}")

    def _add_cameras(self, mjcf_model: mjcf.RootElement) -> None:
        """Add cameras to the model."""
        camera_max_width = 640
        camera_max_height = 480

        for camera in self.cameras:
            direction = np.array(
                [
                    camera.look_at[0] - camera.pos[0],
                    camera.look_at[1] - camera.pos[1],
                    camera.look_at[2] - camera.pos[2],
                ]
            )
            direction = direction / np.linalg.norm(direction)
            up = np.array([0, 0, 1])
            right = np.cross(direction, up)
            right = right / np.linalg.norm(right)
            up = np.cross(right, direction)

            camera_params = {
                "pos": f"{camera.pos[0]} {camera.pos[1]} {camera.pos[2]}",
                "mode": "fixed",
                "fovy": camera.vertical_fov,
                "xyaxes": f"{right[0]} {right[1]} {right[2]} {up[0]} {up[1]} {up[2]}",
            }
            mjcf_model.worldbody.add("camera", name=f"{camera.name}_custom", **camera_params)
            camera_max_width = max(camera_max_width, camera.width)
            camera_max_height = max(camera_max_height, camera.height)

        if camera_max_width > 640 or camera_max_height > 480:
            self._set_framebuffer_size(mjcf_model, camera_max_width, camera_max_height)

    def _get_actuator_states(self, obj_name):
        """Get actuator states (targets and forces)."""
        actuator_states = {
            "dof_pos_target": {},
            "dof_vel_target": {},
            "dof_torque": {},
        }

        # Find the robot index
        robot_idx = None
        for i, robot in enumerate(self.robots):
            if robot.name == obj_name:
                robot_idx = i
                break

        if robot_idx is None:
            return actuator_states

        robot_name = self._mujoco_robot_names[robot_idx]

        for actuator_id in range(self.physics.model.nu):
            actuator = self.physics.model.actuator(actuator_id)
            if actuator.name.startswith(robot_name):
                clean_name = actuator.name[len(robot_name) :]

                actuator_states["dof_pos_target"][clean_name] = float(
                    self.physics.data.ctrl[actuator_id].item()
                )  # Hardcoded to position control
                actuator_states["dof_vel_target"][clean_name] = None
                actuator_states["dof_torque"][clean_name] = float(self.physics.data.actuator_force[actuator_id].item())

        return actuator_states

    def _pack_state(self, body_ids: list[int]):
        """
        Pack pos(3), quat(4), lin_vel_world(3), ang_vel(3) for one-env MuJoCo.

        Args:
            body_ids: list of body IDs, e.g. [root_id] or [root_id] + body_ids_reindex

        Returns:
            root_np: numpy (13,)      — the first body
            body_np: numpy (n_body,13)     — n_body bodies
        """
        data = self.physics.data
        pos = data.xpos[body_ids]
        quat = data.xquat[body_ids]

        # angular ω (world) & v @ subtree_com
        w = data.cvel[body_ids, 0:3]
        v = data.cvel[body_ids, 3:6]

        # compute world‐frame linear velocity at body origin
        offset = data.xpos[body_ids] - data.subtree_com[body_ids]
        lin_world = v + np.cross(w, offset)

        full = np.concatenate([pos, quat, lin_world, w], axis=1)
        root_np = full[0]
        return root_np, full[1:]  # root, bodies

    def _mirror_state_to_native(self):
        self._mj_data = self.physics._data._data

    def _set_root_state(self, obj_name, obj_state, zero_vel=False):
        """Set root position and rotation."""
        if "pos" not in obj_state and "rot" not in obj_state:
            return

        # Check if it's a robot
        robot_idx = None
        for i, robot in enumerate(self.robots):
            if robot.name == obj_name:
                robot_idx = i
                break

        if robot_idx is not None:
            robot = self.robots[robot_idx]
            robot_name = self._mujoco_robot_names[robot_idx]
            if not robot.fix_base_link:
                root_joint = self.physics.data.joint(robot_name)
                root_joint.qpos[:3] = obj_state.get("pos", [0, 0, 0])
                root_joint.qpos[3:7] = obj_state.get("rot", [1, 0, 0, 0])
                if zero_vel:
                    root_joint.qvel[:6] = 0
            else:
                root_body = self.physics.named.model.body_pos[robot_name]
                root_body_quat = self.physics.named.model.body_quat[robot_name]
                root_body[:] = obj_state.get("pos", [0, 0, 0])
                root_body_quat[:] = obj_state.get("rot", [1, 0, 0, 0])
        else:
            model_name = self.mj_objects[obj_name].model + "/"
            try:
                obj_joint = self.physics.data.joint(model_name)
                obj_joint.qpos[:3] = obj_state["pos"]
                obj_joint.qpos[3:7] = obj_state["rot"]
                if zero_vel:
                    obj_joint.qvel[:6] = 0
            except KeyError:
                obj_body = self.physics.named.model.body_pos[model_name]
                obj_body_quat = self.physics.named.model.body_quat[model_name]
                obj_body[:] = obj_state["pos"]
                obj_body_quat[:] = obj_state["rot"]

    def _set_joint_state(self, obj_name, obj_state, zero_vel=False):
        """Set joint positions."""
        if "dof_pos" not in obj_state:
            return

        # Check if it's a robot
        robot_idx = None
        for i, robot in enumerate(self.robots):
            if robot.name == obj_name:
                robot_idx = i
                break

        for joint_name, joint_pos in obj_state["dof_pos"].items():
            if robot_idx is not None:
                robot_name = self._mujoco_robot_names[robot_idx]
                full_joint_name = f"{robot_name}{joint_name}"
            else:
                full_joint_name = f"{obj_name}/{joint_name}"

            joint = self.physics.data.joint(full_joint_name)
            joint.qpos = joint_pos
            if zero_vel:
                joint.qvel = 0
            try:
                actuator = self.physics.model.actuator(full_joint_name)
                self.physics.data.ctrl[actuator.id] = joint_pos
            except KeyError:
                pass

    def _disable_robotgravity(self):
        gravity_vec = np.array(self.scenario.gravity)

        self.physics.data.xfrc_applied[:] = 0
        for body_name in self.robot_body_names:
            body_id = self.physics.model.body(body_name).id
            force_vec = -gravity_vec * self.physics.model.body(body_name).mass
            self.physics.data.xfrc_applied[body_id, 0:3] = force_vec
            self.physics.data.xfrc_applied[body_id, 3:6] = 0

    def refresh_render(self) -> None:
        self.physics.forward()  # Recomputes the forward dynamics without advancing the simulation.
        if self.viewer is not None:
            self.viewer.sync()

    def _get_camera_params(self, camera_id: str, camera):
        """Get camera intrinsics and extrinsics from MuJoCo camera configuration.

        Returns:
            Ks: (3, 3) intrinsic matrix
            c2w: (4, 4) camera-to-world transformation matrix
        """
        mj_camera = self.physics.model.camera(camera_id)

        # Extrinsics: build from camera configuration
        cam_pos = self.physics.data.cam_xpos[mj_camera.id]

        # Compute camera orientation from pos and look_at
        forward = np.array(camera.look_at) - np.array(camera.pos)
        forward = forward / np.linalg.norm(forward)

        world_up = np.array([0, 0, 1])
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, forward)

        # Build c2w matrix (OpenGL convention: camera looks along -Z)
        c2w = np.eye(4)
        c2w[:3, 0] = right
        c2w[:3, 1] = up
        c2w[:3, 2] = -forward  # Z axis points backward
        c2w[:3, 3] = cam_pos

        # Intrinsics: compute from vertical FOV
        fovy_rad = np.deg2rad(camera.vertical_fov)
        fy = camera.height / (2 * np.tan(fovy_rad / 2))
        fx = fy  # assume square pixels
        cx = camera.width / 2.0
        cy = camera.height / 2.0
        Ks = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        return Ks, c2w

    def _apply_scale_to_mjcf(self, mjcf_model, scale):
        """Apply scale to all geoms, bodies, and sites in the MJCF model."""
        scale_x, scale_y, scale_z = scale

        for geom in mjcf_model.find_all("geom"):
            if hasattr(geom, "size") and geom.size is not None:
                size = list(geom.size)
                if geom.type in ["box", None]:
                    if len(size) >= 3:
                        geom.size = [size[0] * scale_x, size[1] * scale_y, size[2] * scale_z]
                elif geom.type == "sphere":
                    if len(size) >= 1:
                        geom.size = [size[0] * max(scale_x, scale_y, scale_z)]
                elif geom.type == "cylinder":
                    if len(size) >= 2:
                        radius_scale = max(scale_x, scale_y)
                        geom.size = [size[0] * radius_scale, size[1] * scale_z]
                elif geom.type == "capsule":
                    if len(size) >= 2:
                        radius_scale = max(scale_x, scale_y)
                        geom.size = [size[0] * radius_scale, size[1] * scale_z]
                elif geom.type == "ellipsoid":
                    if len(size) >= 3:
                        geom.size = [size[0] * scale_x, size[1] * scale_y, size[2] * scale_z]

            if hasattr(geom, "pos") and geom.pos is not None:
                pos = list(geom.pos)
                if len(pos) >= 3:
                    geom.pos = [pos[0] * scale_x, pos[1] * scale_y, pos[2] * scale_z]

        for body in mjcf_model.find_all("body"):
            if hasattr(body, "pos") and body.pos is not None:
                pos = list(body.pos)
                if len(pos) >= 3:
                    body.pos = [pos[0] * scale_x, pos[1] * scale_y, pos[2] * scale_z]

        for site in mjcf_model.find_all("site"):
            if hasattr(site, "pos") and site.pos is not None:
                pos = list(site.pos)
                if len(pos) >= 3:
                    site.pos = [pos[0] * scale_x, pos[1] * scale_y, pos[2] * scale_z]

            if hasattr(site, "size") and site.size is not None:
                size = list(site.size)
                if len(size) >= 1:
                    site.size = [size[0] * max(scale_x, scale_y, scale_z)]

        for joint in mjcf_model.find_all("joint"):
            if hasattr(joint, "pos") and joint.pos is not None:
                pos = list(joint.pos)
                if len(pos) >= 3:
                    joint.pos = [pos[0] * scale_x, pos[1] * scale_y, pos[2] * scale_z]

        # Apply scale to mesh elements (for visual meshes)
        for mesh in mjcf_model.find_all("mesh"):
            if hasattr(mesh, "scale") and mesh.scale is not None:
                mesh_scale = list(mesh.scale)
                if len(mesh_scale) >= 3:
                    mesh.scale = [
                        mesh_scale[0] * scale_x,
                        mesh_scale[1] * scale_y,
                        mesh_scale[2] * scale_z,
                    ]
                elif len(mesh_scale) == 1:
                    # Uniform scale
                    uniform_scale = max(scale_x, scale_y, scale_z)
                    mesh.scale = [mesh_scale[0] * uniform_scale]

    def _set_framebuffer_size(self, mjcf_model, width, height):
        visual_elem = mjcf_model.visual
        global_elem = None
        for child in visual_elem._children:
            if child.tag == "global":
                global_elem = child
                break
        if global_elem is None:
            global_elem = visual_elem.add("global")
        global_elem.offwidth = width
        global_elem.offheight = height

    def _create_builtin_xml(self, obj):
        if isinstance(obj, PrimitiveCubeCfg):
            size_str = f"{obj.half_size[0]} {obj.half_size[1]} {obj.half_size[2]}"
            type_str = "box"
        elif isinstance(obj, PrimitiveCylinderCfg):
            size_str = f"{obj.radius} {obj.height}"
            type_str = "cylinder"
        elif isinstance(obj, PrimitiveSphereCfg):
            size_str = f"{obj.radius}"
            type_str = "sphere"
        else:
            raise ValueError("Unknown primitive type")

        rgba_str = f"{obj.color[0]} {obj.color[1]} {obj.color[2]} 1"
        xml = f"""
        <mujoco model="{obj.name}_model">
        <worldbody>
            <body name="{type_str}_body" pos="{0} {0} {0}">
            <geom name="{type_str}_geom" type="{type_str}" size="{size_str}" rgba="{rgba_str}"/>
            </body>
        </worldbody>
        </mujoco>
        """
        return xml.strip()

    # Properties
    @property
    def actions_cache(self) -> list[ActionType]:
        return self._actions_cache

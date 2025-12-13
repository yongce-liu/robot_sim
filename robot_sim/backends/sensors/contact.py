
@configclass
class ContactSensor(SensorConfig):
    """Contact/force sensor."""

    type: SensorType = SensorType.CONTACT
    """Sensor type, defaults to CONTACT."""
    history_length: int = 3
    """Length of contact force history."""
    _current_contact_force: torch.Tensor | None = None
    """Current contact force."""
    _contact_forces_queue: deque[torch.Tensor] | None = None
    """Queue of contact forces."""

    def _post_init__(self):
        super()._post_init__()
        self._contact_forces_queue = deque(maxlen=self.history_length)

    # def bind_handler(self, handler: BaseSimHandler, *args, **kwargs):
    #     """Bind the simulator handler and pre-compute per-robot indexing."""
    #     super().bind_handler(handler, *args, **kwargs)
    #     self.simulator = handler.scenario.simulator
    #     self.num_envs = handler.scenario.num_envs
    #     self.robots = handler.robots
    #     if self.simulator in ["isaacgym", "mujoco"]:
    #         self.body_ids_reindex = handler._get_body_ids_reindex(self.robots[0].name)
    #     elif self.simulator == "isaacsim":
    #         sorted_body_names = self.handler.get_body_names(self.robots[0].name, True)
    #         self.body_ids_reindex = torch.tensor(
    #             [self.handler.contact_sensor.body_names.index(name) for name in sorted_body_names],
    #             dtype=torch.int,
    #             device=self.handler.device,
    #         )
    #     else:
    #         raise NotImplementedError
    #     self.initialize()
    #     self.__call__()

    # def initialize(self):
    #     """Warm-start the queue with `history_length` entries."""
    #     for _ in range(self.history_length):
    #         if self.simulator == "isaacgym":
    #             self._current_contact_force = isaacgym.gymtorch.wrap_tensor(
    #                 self.handler.gym.acquire_net_contact_force_tensor(self.handler.sim)
    #             )
    #         elif self.simulator == "isaacsim":
    #             self._current_contact_force = self.handler.contact_sensor.data.net_forces_w
    #         elif self.simulator == "mujoco":
    #             self._current_contact_force = self._get_contact_forces_mujoco()
    #         else:
    #             raise NotImplementedError
    #         self._contact_forces_queue.append(
    #             self._current_contact_force.clone().view(self.num_envs, -1, 3)[:, self.body_ids_reindex, :]
    #         )

    # def _get_contact_forces_mujoco(self) -> torch.Tensor:
    #     """Compute net contact forces on each body.

    #     Returns:
    #         torch.Tensor: shape (nbody, 3), contact forces for each body
    #     """
    #     nbody = self.handler.physics.model.nbody
    #     contact_forces = torch.zeros((nbody, 3), device=self.handler.device)

    #     for i in range(self.handler.physics.data.ncon):
    #         contact = self.handler.physics.data.contact[i]
    #         force = np.zeros(6, dtype=np.float64)
    #         mujoco.mj_contactForce(self.handler.physics.model.ptr, self.handler.physics.data.ptr, i, force)
    #         f_contact = torch.from_numpy(force[:3]).to(device=self.handler.device)

    #         body1 = self.handler.physics.model.geom_bodyid[contact.geom1]
    #         body2 = self.handler.physics.model.geom_bodyid[contact.geom2]

    #         contact_forces[body1] += f_contact
    #         contact_forces[body2] -= f_contact

    #     return contact_forces

    # def __call__(self):
    #     """Fetch the newest net contact forces and update the queue."""
    #     if self.simulator == "isaacgym":
    #         self.handler.gym.refresh_net_contact_force_tensor(self.handler.sim)
    #     elif self.simulator == "isaacsim":
    #         self._current_contact_force = self.handler.contact_sensor.data.net_forces_w
    #     elif self.simulator == "mujoco":
    #         self._current_contact_force = self._get_contact_forces_mujoco()
    #     else:
    #         raise NotImplementedError
    #     self._contact_forces_queue.append(
    #         self._current_contact_force.view(self.num_envs, -1, 3)[:, self.body_ids_reindex, :]
    #     )
    #     return {self.robots[0].name: self}

    # @property
    # def contact_forces_history(self) -> torch.Tensor:
    #     """Return stacked history as (num_envs, history_length, num_bodies, 3)."""
    #     return torch.stack(list(self._contact_forces_queue), dim=1)  # (num_envs, history_length, num_bodies, 3)

    # @property
    # def contact_forces(self) -> torch.Tensor:
    #     """Return the latest contact forces snapshot."""
    #     return self._contact_forces_queue[-1]


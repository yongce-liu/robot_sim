"""Simulation manager for single or multi-backend simulation with multiprocessing support."""

from abc import ABC


class SimulationManager(ABC):
    pass


#     """Manager for single or multi-backend simulation with multiprocessing support.

#     This class provides a unified interface to run simulations with one or multiple
#     backends simultaneously. Each backend runs in its own process for true parallelism
#     and isolation. The manager communicates with backend processes through queues.

#     Features:
#     - True parallel execution with multiprocessing
#     - Process isolation - one backend crash won't affect others
#     - Support for GPU parallelism across processes
#     - Unified interface for single or multi-backend scenarios
#     """

#     def __init__(self, config: DictConfig) -> None:
#         """Initialize simulation manager.

#         Args:
#             config: Hydra configuration containing backend settings
#             use_multiprocessing: If True, run each backend in separate process.
#                                 If False, run all backends in main process (legacy mode)
#         """
#         self.config = config

#         # Process management
#         if self._is_setup:
#             raise RuntimeError("Cannot add backends after setup()")

#         backend_config = config if config is not None else self.config

#         if self.use_multiprocessing:
#             # Create communication channels for this backend process
#             self.cmd_queues[name] = Queue()
#             if self.result_queue is None:
#                 self.result_queue = Queue()
#             self.ready_events[name] = Event()
#             self.stop_events[name] = Event()

#             # Convert config to dict for serialization
#             config_dict = OmegaConf.to_container(backend_config, resolve=True)

#             # Create process (but don't start yet)
#             process = Process(
#                 target=_backend_worker,
#                 args=(
#                     name,
#                     backend_type,
#                     config_dict,
#                     self.cmd_queues[name],
#                     self.result_queue,
#                     self.ready_events[name],
#                     self.stop_events[name],
#                 ),
#                 name=f"Backend-{name}",
#             )
#             self.processes[name] = process
#             print(f"Added backend: {name} ({backend_type}) [multiprocessing mode]")
#         else:
#             if not self.processes:
#                 raise RuntimeError("No backends added. Use add_backend() first.")

#             print(f"Setting up {len(self.processes)} backend(s) in separate processes...")

#             # Start all backend processes
#             for name, process in self.processes.items():
#                 print(f"  Starting process for {name} (PID will be assigned)...")
#                 process.start()

#         _send_command(self, name: str, cmd_type: str, **kwargs) -> int:
#         """Send command to a backend process.

#         Args:
#             name: Backend name
#             cmd_type: Command type
#             **kwargs: Additional command parameters

#         Returns:
#             Command ID for tracking response
#         """
#         cmd_id = self._cmd_id_counter
#         self._cmd_id_counter += 1

#         cmd = {"type": cmd_type, "id": cmd_id, **kwargs}
#         self.cmd_queues[name].put(cmd)
#         return cmd_id

#     def _collect_results(self, expected_count: int, timeout: float = 5.0) -> dict[str, any]:
#         """Collect results from backend processes.

#         Args:
#             expected_count: Number of results to collect
#             timeout: Maximum time to wait for all results

#         Returns:
#             dict mapping backend names to their results
#         """
#         results = {}
#         for _ in range(expected_count):
#             try:
#                 result = self.result_queue.get(timeout=timeout)
#                 if "error" in result:
#                     raise RuntimeError(f"Backend {result['name']} error: {result['error']}")
#                 results[result["name"]] = result["result"]
#             except Empty:
#                 raise TimeoutError(f"Timeout waiting for backend results")
#         return results

#     def step(self) -> dict[str, dict[str, any]]:
#         """Step all backends forward by one timestep.

#         Returns:
#             dict mapping backend names to their step results
#         """
#         if not self._is_setup:
#             raise RuntimeError("Manager not setup. Call setup() first.")

#         if self.use_multiprocessing:
#             # Send step command to all backends
#             for name in self.processes.keys():
#                 self._send_command(name, "step")

#             # Collect results from all backends
#             return self._collect_results(len(self.processes))
#         if self.use_multiprocessing:
#             # Send reset command to all backends
#             for name in self.processes.keys():
#                 self._send_command(name, "reset")

#             # Collect results from all backends
#             return self._collect_results(len(self.processes))
#         else: and terminate processes."""
#         print("Closing all backends...")

#         if self.use_multiprocessing:
#             # Send close command to all backend processes
#             for name in self.processes.keys():
#                 print(f"  Closing {name}...")
#                 self._send_command(name, "close")
#                 self.stop_events[name].set()

#             # Wait for all results
#             try:
#                 self._collect_results(len(self.processes), timeout=5.0)
#             except (TimeoutError, Empty):
#                 print("Warning: Some backends did not respond to close command")

#             # Join all processes
#             for name, process in self.processes.items():
#                 process.join(timeout=3.0)
#                 if process.is_alive():
#                     print(f"  Force terminating {name}...")
#                     process.terminate()
#                     process.join(timeout=1.0)
#                 print(f"  ✓ {name} closed")

#             # Clear data structures
#             self.processes.clear()
#             self.cmd_queues.clear()
#             self.ready_events.clear()
#             self.stop_events.clear()
#             self.result_queue = None
#         else:
#             # Legacy single-process mode
#             for name, backend in self.backends.items():
#                 print(f"  Closing {name}...")
#                     results[name] = backend.reset()
#            self.use_multiprocessing:
#             if isinstance(action, dict):
#                 # Send specific actions to each backend
#                 for name, act in action.items():
#                     if name in self.processes:
#                         self._send_command(name, "apply_action", action=act)
#             else:
#                 # Apply same action to all backends
#                 for name in self.processes.keys():
#                     self._send_command(name, "apply_action", action=action)

#             # Collect acknowledgments
#             self._collect_results(len(self.processes) if not isinstance(action, dict) else len(action))
#         else:
#             # Legacy single-process mode
#         if self.use_multiprocessing:
#             # Send get_observation command to all backends
#             for name in self.processes.keys():
#                 self._send_command(name, "get_observation")

#             # Collect observations from all backends
#             return self._collect_results(len(self.processes))
#         else:
#             # Legacy single-process mode
#                 if isinstance(action, dict):
#                 for name, act in action.items():
#                     if name in self.backends:
#                         self.backends[name].apply_action(act)
#             else:
#                 # Apply same action to all backends
#                 for backend in self.backends.values():
#         def add_backend(self, name: str, backend_type: str, config: Optional[DictConfig] = None) -> None:
#         """Add a backend to the simulation.

#         Args:
#             name: Identifier for this backend instance
#             backend_type: Type of backend ("isaac", "mujoco", etc.)
#             config: Optional configuration for this specific backend
#         """
#         backend_config = config if config is not None else self.config

#         if backend_type.lower() == "isaac":
#             self.backends[name] = IsaacBackend(backend_config)
#         elif backend_type.lower() == "mujoco":
#             self.backends[name] = MuJoCoBackend(backend_config)
#         else:
#             raise ValueError(f"Unknown backend type: {backend_type}")

#         print(f"Added backend: {name} ({backend_type})")

#     def setup(self) -> None:
#         """Setup all backends."""
#         if not self.backends:
#             raise RuntimeError("No backends added. Use add_backend() first.")

#         print(f"Setting up {len(self.backends)} backend(s)...")
#         for name, backend in self.backends.items():
#             print(f"  Setting up {name}...")
#             backend.setup()

#         self._is_setup = True
#         print("All backends initialized successfully!")

#     def step(self) -> dict[str, dict[str, any]]:
#         """Step all backends forward by one timestep.

#         Returns:
#             dict mapping backend names to their step results
#         """
#         if not self._is_setup:
#         if self.use_multiprocessing:
#             # Send get_state command to all backends
#             for name in self.processes.keys():
#                 self._send_command(name, "get_state")

#             # Collect states from all backends
#             return self._collect_results(len(self.processes))
#         else:
#             # Legacy single-process mode
#                 raise RuntimeError("Manager not setup. Call setup() first.")

#         resuelf.use_multiprocessing:
#             if source not in self.processes:
#                 raise ValueError(f"Source backend '{source}' not found")

#             # Get state from source backend
#             self._send_command(source, "get_state")
#             results = self._collect_results(1)
#             source_state = results[source]

#             # Determine targets
#             if targets is None:
#                 targets = [name for name in self.processes.keys() if name != source]

#             # Set state on target backends
#             for target in targets:
#                 if target in self.processes:
#                     self._send_command(target, "set_state", state=source_state)
#                     print(f"Synchronized state: {source} -> {target}")

#             # Collect acknowledgments
#             if targets:
#                 self._collect_results(len(targets))
#         else:
#             # Legacy single-process mode
#             if source not in self.backends:
#                 raise ValueError(f"Source backend '{source}' not found")

#             source_state = self.backends[source].get_state()

#             if targets is None:
#                 targets = [name for name in self.backends.keys() if name != source]

#             for target in targets:
#                 if target in self.backends:
#                     self.backends[target].set_state(source_state)
#             """
#         if not self._is_setup:
#             raise RuntimeError("Manager not setup. Call setup() first.")

#         results = {}
#         for name, backend in self.backends.items():
#             results[name] = backend.reset()

#         return results

#     def close(self) -> None:
#         """Close all backends."""
#         print("Closing all backends...")
#         for name, backend in self.backends.items():
#             print(f"  Closing {name}...")
#             backend.close()

#         self._is_setup = False
#         print("All backends closed.")

#     def apply_action(selprocesses):
#         if self.use_multiprocessing:
#             pass
#         else:
#             len(self.backends)

#     @property
#     def backend_names(self) -> list[str]:
#         """Get list of backend names."""
#         return list(self.processes.keys()) if self.use_multiprocessing else list(self.backends.keys())

#     @property
#     def process_info(self) -> dict[str, dict[str, any]]:
#         """Get information about backend processes.

#         Returns:
#             dict mapping backend names to their process information (PID, alive status, etc.)
#         """
#         if not self.use_multiprocessing:
#             return {"mode": "single-process"}

#         info = {}
#         for name, process in self.processes.items():
#             info[name] = {
#                 "pid": process.pid,
#                 "alive": process.is_alive(),
#                 "exitcode": process.exitcode,
#             }
#         return info

#     def __repr__(self) -> str:
#         """String representation."""
#         mode = "multiprocessing" if self.use_multiprocessing else "single-process"
#         if self.use_multiprocessing:
#             backends_str = ", ".join(f"{name}(PID:{p.pid})" for name, p in self.processes.items())
#         else:
#             backends_str = ", ".join(f"{name}({backend.backend_name})" for name, backend in self.backends.items())
#         return f"SimulationManager[{mode}]({backends_str})"

#     def __del__(self):
#         """Cleanup on deletion."""
#         if self._is_setup:
#             try:
#                 self.close()
#             except Exception:
#                 pass  # Best effort cleanupn(act)
#         else:
#             # Apply same action to all backends
#             for backend in self.backends.values():
#                 backend.apply_action(action)

#     def get_observations(self) -> dict[str, np.ndarray]:
#         """Get observations from all backends.

#         Returns:
#             dict mapping backend names to their observations
#         """
#         return {name: backend.get_observation() for name, backend in self.backends.items()}

#     def get_states(self) -> dict[str, dict[str, any]]:
#         """Get states from all backends.

#         Returns:
#             dict mapping backend names to their states
#         """
#         return {name: backend.get_state() for name, backend in self.backends.items()}

#     def synchronize_states(self, source: str, targets: Optional[list[str]] = None) -> None:
#         """Synchronize state from source backend to target backends.

#         This is useful for joint simulation to ensure consistency.

#         Args:
#             source: Name of source backend
#             targets: list of target backend names (None = all others)
#         """
#         if source not in self.backends:
#             raise ValueError(f"Source backend '{source}' not found")

#         source_state = self.backends[source].get_state()

#         if targets is None:
#             targets = [name for name in self.backends.keys() if name != source]

#         for target in targets:
#             if target in self.backends:
#                 self.backends[target].set_state(source_state)
#                 print(f"Synchronized state: {source} -> {target}")

#     def compare_states(self) -> dict[str, any]:
#         """Compare states across all backends.

#         Returns:
#             dict containing comparison metrics
#         """
#         if len(self.backends) < 2:
#             print("Warning: Need at least 2 backends for comparison")
#             return {}

#         states = self.get_states()
#         backend_names = list(states.keys())

#         # Compare joint positions
#         differences = {}
#         ref_name = backend_names[0]
#         ref_state = states[ref_name]

#         for name in backend_names[1:]:
#             state = states[name]

#             # Calculate position difference
#             pos_diff = np.linalg.norm(ref_state["joint_positions"] - state["joint_positions"])

#             differences[f"{ref_name}_vs_{name}"] = {
#                 "joint_position_diff": pos_diff,
#                 "base_position_diff": np.linalg.norm(ref_state["base_position"] - state["base_position"]),
#             }

#         return differences

#     @property
#     def num_backends(self) -> int:
#         """Get number of backends."""
#         return len(self.backends)

#     @property
#     def backend_names(self) -> list[str]:
#         """Get list of backend names."""
#         return list(self.backends.keys())

#     def __repr__(self) -> str:
#         """String representation."""
#         backends_str = ", ".join(f"{name}({backend.backend_name})" for name, backend in self.backends.items())
#         return f"SimulationManager({backends_str})"


# ############# Helper functions for multiprocessing backend worker #############
# def _backend_worker(
#     name: str,
#     backend_type: str,
#     config_dict: dict,
#     cmd_queue: Queue,
#     result_queue: Queue,
#     ready_event: Event,
#     stop_event: Event,
# ) -> None:
#     """Worker process for running a backend simulator.

#     Args:
#         name: Backend identifier
#         backend_type: Type of backend ("isaac", "mujoco")
#         config_dict: Configuration dictionary (serialized OmegaConf)
#         cmd_queue: Queue for receiving commands from manager
#         result_queue: Queue for sending results back to manager
#         ready_event: Event to signal when backend is ready
#         stop_event: Event to signal process should stop
#     """
#     try:
#         # Reconstruct config from dict
#         config = OmegaConf.create(config_dict)

#         # Create backend instance in this process
#         if backend_type.lower() == "isaac":
#             backend = IsaacBackend(config)
#         elif backend_type.lower() == "mujoco":
#             backend = MuJoCoBackend(config)
#         else:
#             raise ValueError(f"Unknown backend type: {backend_type}")

#         # Setup backend
#         backend.setup()
#         ready_event.set()  # Signal that backend is ready

#         print(f"[{name}] Backend worker started in process {mp.current_process().pid}")

#         # Command processing loop
#         while not stop_event.is_set():
#             try:
#                 # Get command with timeout
#                 cmd = cmd_queue.get(timeout=0.1)

#                 cmd_type = cmd.get("type")
#                 cmd_id = cmd.get("id")

#                 if cmd_type == "step":
#                     result = backend.step()
#                     result_queue.put({"id": cmd_id, "name": name, "result": result})

#                 elif cmd_type == "reset":
#                     result = backend.reset()
#                     result_queue.put({"id": cmd_id, "name": name, "result": result})

#                 elif cmd_type == "apply_action":
#                     action = cmd.get("action")
#                     backend.apply_action(action)
#                     result_queue.put({"id": cmd_id, "name": name, "result": "ok"})

#                 elif cmd_type == "get_observation":
#                     obs = backend.get_observation()
#                     result_queue.put({"id": cmd_id, "name": name, "result": obs})

#                 elif cmd_type == "get_state":
#                     state = backend.get_state()
#                     result_queue.put({"id": cmd_id, "name": name, "result": state})

#                 elif cmd_type == "set_state":
#                     state = cmd.get("state")
#                     backend.set_state(state)
#                     result_queue.put({"id": cmd_id, "name": name, "result": "ok"})

#                 elif cmd_type == "close":
#                     backend.close()
#                     result_queue.put({"id": cmd_id, "name": name, "result": "closed"})
#                     break

#             except Empty:
#                 continue
#             except Exception as e:
#                 result_queue.put({"id": cmd_id, "name": name, "error": str(e)})
#                 print(f"[{name}] Error in worker: {e}")

#     except Exception as e:
#         print(f"[{name}] Fatal error in backend worker: {e}")
#         ready_event.set()  # Unblock manager even on error
#     finally:
#         print(f"[{name}] Backend worker stopped")

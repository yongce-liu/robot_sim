"""Simulation manager for single or multi-backend simulation."""

from typing import Any, Dict, List, Optional, Union

import numpy as np
from omegaconf import DictConfig

from robot_sim.backend.base import BaseBackend
from robot_sim.backend.isaac import IsaacBackend
from robot_sim.backend.mujoco import MuJoCoBackend


class SimulationManager:
    """Manager for single or multi-backend simulation.
    
    This class provides a unified interface to run simulations with one or multiple
    backends simultaneously, enabling easy switching between simulators or running
    joint simulations for comparison/validation.
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialize simulation manager.
        
        Args:
            config: Hydra configuration containing backend settings
        """
        self.config = config
        self.backends: Dict[str, BaseBackend] = {}
        self._is_setup = False

    def add_backend(self, name: str, backend_type: str, config: Optional[DictConfig] = None) -> None:
        """Add a backend to the simulation.
        
        Args:
            name: Identifier for this backend instance
            backend_type: Type of backend ("isaac", "mujoco", etc.)
            config: Optional configuration for this specific backend
        """
        backend_config = config if config is not None else self.config
        
        if backend_type.lower() == "isaac":
            self.backends[name] = IsaacBackend(backend_config)
        elif backend_type.lower() == "mujoco":
            self.backends[name] = MuJoCoBackend(backend_config)
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")
        
        print(f"Added backend: {name} ({backend_type})")

    def setup(self) -> None:
        """Setup all backends."""
        if not self.backends:
            raise RuntimeError("No backends added. Use add_backend() first.")
        
        print(f"Setting up {len(self.backends)} backend(s)...")
        for name, backend in self.backends.items():
            print(f"  Setting up {name}...")
            backend.setup()
        
        self._is_setup = True
        print("All backends initialized successfully!")

    def step(self) -> Dict[str, Dict[str, Any]]:
        """Step all backends forward by one timestep.
        
        Returns:
            Dict mapping backend names to their step results
        """
        if not self._is_setup:
            raise RuntimeError("Manager not setup. Call setup() first.")
        
        results = {}
        for name, backend in self.backends.items():
            results[name] = backend.step()
        
        return results

    def reset(self) -> Dict[str, Dict[str, Any]]:
        """Reset all backends.
        
        Returns:
            Dict mapping backend names to their reset results
        """
        if not self._is_setup:
            raise RuntimeError("Manager not setup. Call setup() first.")
        
        results = {}
        for name, backend in self.backends.items():
            results[name] = backend.reset()
        
        return results

    def close(self) -> None:
        """Close all backends."""
        print("Closing all backends...")
        for name, backend in self.backends.items():
            print(f"  Closing {name}...")
            backend.close()
        
        self._is_setup = False
        print("All backends closed.")

    def apply_action(self, action: Union[np.ndarray, Dict[str, np.ndarray]]) -> None:
        """Apply action to backend(s).
        
        Args:
            action: Either a single action array (applied to all backends) or
                   a dict mapping backend names to their specific actions
        """
        if isinstance(action, dict):
            for name, act in action.items():
                if name in self.backends:
                    self.backends[name].apply_action(act)
        else:
            # Apply same action to all backends
            for backend in self.backends.values():
                backend.apply_action(action)

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations from all backends.
        
        Returns:
            Dict mapping backend names to their observations
        """
        return {
            name: backend.get_observation()
            for name, backend in self.backends.items()
        }

    def get_states(self) -> Dict[str, Dict[str, Any]]:
        """Get states from all backends.
        
        Returns:
            Dict mapping backend names to their states
        """
        return {
            name: backend.get_state()
            for name, backend in self.backends.items()
        }

    def synchronize_states(self, source: str, targets: Optional[List[str]] = None) -> None:
        """Synchronize state from source backend to target backends.
        
        This is useful for joint simulation to ensure consistency.
        
        Args:
            source: Name of source backend
            targets: List of target backend names (None = all others)
        """
        if source not in self.backends:
            raise ValueError(f"Source backend '{source}' not found")
        
        source_state = self.backends[source].get_state()
        
        if targets is None:
            targets = [name for name in self.backends.keys() if name != source]
        
        for target in targets:
            if target in self.backends:
                self.backends[target].set_state(source_state)
                print(f"Synchronized state: {source} -> {target}")

    def compare_states(self) -> Dict[str, Any]:
        """Compare states across all backends.
        
        Returns:
            Dict containing comparison metrics
        """
        if len(self.backends) < 2:
            print("Warning: Need at least 2 backends for comparison")
            return {}
        
        states = self.get_states()
        backend_names = list(states.keys())
        
        # Compare joint positions
        differences = {}
        ref_name = backend_names[0]
        ref_state = states[ref_name]
        
        for name in backend_names[1:]:
            state = states[name]
            
            # Calculate position difference
            pos_diff = np.linalg.norm(
                ref_state["joint_positions"] - state["joint_positions"]
            )
            
            differences[f"{ref_name}_vs_{name}"] = {
                "joint_position_diff": pos_diff,
                "base_position_diff": np.linalg.norm(
                    ref_state["base_position"] - state["base_position"]
                ),
            }
        
        return differences

    @property
    def num_backends(self) -> int:
        """Get number of backends."""
        return len(self.backends)

    @property
    def backend_names(self) -> List[str]:
        """Get list of backend names."""
        return list(self.backends.keys())

    def __repr__(self) -> str:
        """String representation."""
        backends_str = ", ".join(
            f"{name}({backend.backend_name})"
            for name, backend in self.backends.items()
        )
        return f"SimulationManager({backends_str})"

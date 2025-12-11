"""Basic simulation example using Hydra configuration."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from robot_sim.backend.mujoco import MuJoCoBackend


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """Run basic simulation example.
    
    Args:
        cfg: Hydra configuration
    """
    print("Basic simulation example")
    print(f"Backend: {cfg.simulation.backend}")
    
    # Initialize backend
    backend = MuJoCoBackend()
    
    # TODO: Add simulation setup and loop
    # Setup
    backend.setup()
    
    # Simulation loop
    for _ in range(cfg.simulation.num_steps):
        backend.step()
    
    # Cleanup
    backend.close()


if __name__ == "__main__":
    main()

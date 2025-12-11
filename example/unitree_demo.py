"""Unitree robot simulation demo using Hydra configuration."""

import hydra
from omegaconf import DictConfig
from robot_sim.backend.isaac import IsaacBackend
from robot_sim.presets.unitree.go2 import Go2Config


@hydra.main(version_base=None, config_path="../configs", config_name="unitree_go2")
def main(cfg: DictConfig) -> None:
    """Run Unitree Go2 simulation demo.

    Args:
        cfg: Hydra configuration
    """
    print(f"Unitree {cfg.robot.type.upper()} simulation demo")
    print(f"Backend: {cfg.simulation.backend}")

    # Load robot configuration
    robot_config = Go2Config()

    # Initialize backend
    backend = IsaacBackend()

    # TODO: Add robot setup and simulation
    # Setup
    backend.setup()

    # Simulation loop
    for _ in range(cfg.simulation.num_steps):
        backend.step()

    # Cleanup
    backend.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Script to run simulations with different configurations."""

import hydra
from omegaconf import DictConfig, OmegaConf
from robot_sim.backend import SimulationManager


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """Main entry point.

    Switch backends easily:
        python scripts/run_sim.py backend=isaac
        python scripts/run_sim.py backend=mujoco
        python scripts/run_sim.py backend=isaac robot=h1

    Args:
        cfg: Hydra configuration object
    """
    # Print configuration
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))

    print(f"\nRunning simulation with backend: {cfg.simulation.backend}")
    print(f"Robot: {cfg.robot.type}")

    # Create simulation manager
    manager = SimulationManager(cfg)

    # Add backend from configuration
    manager.add_backend(name="main", backend_type=cfg.simulation.backend, config=cfg)

    # Setup and run simulation
    manager.setup()

    print(f"\nRunning {cfg.simulation.num_steps} steps...")
    for step in range(cfg.simulation.num_steps):
        results = manager.step()

        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}/{cfg.simulation.num_steps}")

    # Cleanup
    manager.close()
    print("\nSimulation complete!")


if __name__ == "__main__":
    main()

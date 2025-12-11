"""Example: Single backend simulation with easy switching."""

import hydra
from omegaconf import DictConfig
from robot_sim.backend import SimulationManager


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """Run single backend simulation.

    Switch backends using:
        python example/single_backend.py backend=isaac
        python example/single_backend.py backend=mujoco

    Args:
        cfg: Hydra configuration
    """
    print("=" * 60)
    print("Single Backend Simulation")
    print(f"Backend: {cfg.simulation.backend}")
    print(f"Robot: {cfg.robot.type}")
    print("=" * 60)

    # Create simulation manager
    manager = SimulationManager(cfg)

    # Add the backend specified in config
    manager.add_backend(name="main", backend_type=cfg.simulation.backend, config=cfg)

    # Setup simulation
    manager.setup()

    # Run simulation loop
    num_steps = cfg.simulation.num_steps
    print(f"\nRunning {num_steps} simulation steps...")

    for step in range(num_steps):
        # Apply some action (placeholder)
        # action = controller.compute(...)
        # manager.apply_action(action)

        # Step simulation
        results = manager.step()

        # Print progress
        if (step + 1) % 100 == 0:
            print(f"Step {step + 1}/{num_steps}")

    # Get final state
    final_states = manager.get_states()
    print(f"\nFinal state: {final_states['main']}")

    # Cleanup
    manager.close()
    print("\nSimulation complete!")


if __name__ == "__main__":
    main()

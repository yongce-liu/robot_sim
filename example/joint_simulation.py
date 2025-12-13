"""Example: Joint simulation with multiple backends running simultaneously."""

import hydra
from omegaconf import DictConfig

from robot_sim.backends import SimulationManager


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """Run joint simulation with multiple backends.

    This allows comparing different simulators side-by-side or validating
    simulation results across platforms.

    Args:
        cfg: Hydra configuration
    """
    print("=" * 60)
    print("Joint Multi-Backend Simulation")
    print("Running Isaac Lab and MuJoCo simultaneously")
    print("=" * 60)

    # Create simulation manager
    manager = SimulationManager(cfg)

    # Add multiple backends
    print("\nAdding backends...")
    manager.add_backend(name="isaac", backend_type="isaac", config=cfg)
    manager.add_backend(name="mujoco", backend_type="mujoco", config=cfg)

    print(f"Total backends: {manager.num_backends}")
    print(f"Backends: {manager.backend_names}")

    # Setup all backends
    manager.setup()

    # Reset all backends to ensure same initial state
    print("\nResetting all backends...")
    initial_states = manager.reset()  # noqa: F841

    # Synchronize initial states (optional)
    # manager.synchronize_states(source="mujoco", targets=["isaac"])

    # Run simulation loop
    num_steps = min(cfg.simulation.num_steps, 1000)  # Limit for demo
    print(f"\nRunning {num_steps} simulation steps...")

    comparison_interval = 100  # Compare states every N steps

    for step in range(num_steps):
        # Apply same action to both backends
        # action = np.zeros(12)  # Placeholder action
        # manager.apply_action(action)

        # Step all backends
        results = manager.step()  # noqa: F841

        # Compare states periodically
        if (step + 1) % comparison_interval == 0:
            print(f"\n--- Step {step + 1}/{num_steps} ---")

            # Get observations from all backends
            observations = manager.get_observations()
            print(f"Isaac observation shape: {observations['isaac'].shape}")
            print(f"MuJoCo observation shape: {observations['mujoco'].shape}")

            # Compare states
            differences = manager.compare_states()
            if differences:
                print("\nState differences:")
                for comparison, metrics in differences.items():
                    print(f"  {comparison}:")
                    for metric, value in metrics.items():
                        print(f"    {metric}: {value:.6f}")

    # Final comparison
    print("\n" + "=" * 60)
    print("Final State Comparison")
    print("=" * 60)

    final_states = manager.get_states()
    for name, state in final_states.items():
        print(f"\n{name.upper()}:")
        print(f"  Base position: {state['base_position']}")
        print(f"  Joint positions: {state['joint_positions'][:4]}... (showing first 4)")

    final_differences = manager.compare_states()
    if final_differences:
        print("\nFinal differences:")
        for comparison, metrics in final_differences.items():
            print(f"  {comparison}: {metrics}")

    # Cleanup
    manager.close()
    print("\nJoint simulation complete!")


if __name__ == "__main__":
    main()

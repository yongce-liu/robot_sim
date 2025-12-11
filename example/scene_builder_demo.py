"""Example: Using scene builder for quick setup."""

import hydra
from omegaconf import DictConfig
from robot_sim.utils.scene_builder import (
    SceneBuilder,
    create_empty_scene,
    create_obstacle_scene,
    create_stairs_scene,
)


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """Demonstrate scene building utilities.

    Args:
        cfg: Hydra configuration
    """
    print("=" * 60)
    print("Scene Builder Example")
    print("=" * 60)

    # Create custom scene
    print("\n1. Building custom scene...")
    scene = (
        SceneBuilder()
        .add_ground_plane(size=50.0, friction=1.0)
        .add_box(
            name="target",
            position=(5.0, 0, 0.5),
            size=(1.0, 1.0, 1.0),
            mass=10.0,
            color=(0.2, 0.8, 0.2),
        )
        .add_sphere(
            name="ball",
            position=(2.0, 0, 1.0),
            radius=0.3,
            mass=1.0,
            color=(0.8, 0.2, 0.2),
        )
        .add_obstacle_course(num_obstacles=5, spacing=2.0)
    )

    scene_config = scene.build()
    print(f"Created scene with {len(scene_config['objects'])} objects")

    # Use preset scenes
    print("\n2. Using preset scenes...")

    # Empty scene
    empty = create_empty_scene()
    print(f"Empty scene: {len(empty.build()['objects'])} objects")

    # Obstacle course
    obstacles = create_obstacle_scene()
    print(f"Obstacle scene: {len(obstacles.build()['objects'])} objects")

    # Stairs
    stairs = create_stairs_scene()
    print(f"Stairs scene: {len(stairs.build()['objects'])} objects")

    # You can now pass scene_config to backend for loading
    print("\n3. Scene configuration:")
    print(f"Objects: {[obj['name'] for obj in scene_config['objects'][:5]]}...")

    # Example: Initialize simulation with scene
    # manager = SimulationManager(cfg)
    # manager.add_backend("main", cfg.simulation.backend, cfg)
    # manager.setup()
    # Load scene into backend...

    print("\nScene building complete!")


if __name__ == "__main__":
    main()

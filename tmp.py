from robot_sim.configs import configclass


@configclass
class SceneConfig:
    path: str | None = None
    """Path to the scene file (e.g., MJCF file)."""

    def __post_init__(self) -> None:
        pass

scene = SceneConfig()
print(scene.path)
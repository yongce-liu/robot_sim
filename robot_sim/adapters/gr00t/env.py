from robot_sim.envs import MapEnv

from .controller import DecoupledWBCPolicy


class Gr00tWBCEnv(MapEnv):
    """Gr00t Environment Wrapper.

    This environment wrapper is specifically designed for the Gr00t robot simulation.
    It extends the MapEnv class to provide additional functionalities and configurations
    tailored for Gr00t simulations.
    """

    def _init_controller(self) -> DecoupledWBCPolicy:
        return None

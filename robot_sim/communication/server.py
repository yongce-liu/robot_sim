"""Example: Running simulation with model client."""

import hydra
from omegaconf import DictConfig

from robot_sim.backend import SimulationManager
from robot_sim.server.sim_server import SimulationServer
from robot_sim.utils.sensors import IMU, Camera, SensorManager


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    """Run simulation server that communicates with external models.
    
    Usage:
        Terminal 1: python example/model_server.py
        Terminal 2: python example/model_client_example.py
    
    Args:
        cfg: Hydra configuration
    """
    print("=" * 60)
    print("Simulation Server for Model Communication")
    print("=" * 60)
    
    # Create simulation manager
    manager = SimulationManager(cfg)
    manager.add_backend("main", cfg.simulation.backend, cfg)
    manager.setup()
    
    # Setup sensors
    sensor_mgr = SensorManager()
    sensor_mgr.add_sensor(Camera(name="front_camera", width=640, height=480))
    sensor_mgr.add_sensor(IMU(name="imu"))
    
    # Create server
    server = SimulationServer(
        sim_manager=manager,
        sensor_manager=sensor_mgr,
        port=5555,
    )
    
    print("\nServer ready. Waiting for model client connection...")
    print("Start your model client now!")
    
    # Run server
    try:
        server.run(num_steps=cfg.simulation.num_steps)
    except KeyboardInterrupt:
        print("\nServer interrupted")
    finally:
        manager.close()


if __name__ == "__main__":
    main()

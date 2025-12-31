from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from examples.twist2.xrobot_teleop_to_robot_w_hand import XRobotTeleopToRobot
from robot_sim.adapters.twist2 import Twist2Env, Twist2Policy
from robot_sim.configs import SimulatorConfig
from robot_sim.utils.helper import setup_logger

PROJECT_DIR = Path(__file__).parents[0].resolve()


def teleop_worker(teleop_client: XRobotTeleopToRobot) -> None:
    """Worker function to run teleoperation client in a separate process.

    Args:
        teleop_client: XRobotTeleopToRobot instance
    """
    try:
        teleop_client.run()
    except Exception as e:
        logger.exception(f"An error occurred in teleop_worker: {e}")
        raise e


def run(cfg: DictConfig) -> None:
    """Run Gr00t simulation.

    Args:
        cfg: Hydra configuration containing simulator_config, observation_mapping, action_mapping
    """
    setup_logger(f"{HydraConfig.get().runtime.output_dir}/{HydraConfig.get().job.name}.loguru.log")
    logger.info("Starting Twist2 simulation...")
    cfg = hydra.utils.instantiate(cfg, _recursive=True)

    # Initialize Twist2Env
    logger.info("Initializing Twist2Env...")

    # # Start teleoperation and simulation workers
    # import time
    # from multiprocessing import Process
    # teleop_process = Process(target=teleop_worker, args=(cfg.teleop,))
    # teleop_process.start()
    # teleop_process.join()
    # time.sleep(10)  # Ensure teleop client is ready

    policy: Twist2Policy = cfg.policy
    env = Twist2Env(config=SimulatorConfig.from_dict(OmegaConf.to_container(cfg.simulator, resolve=True)))

    num_steps = 10000
    obs, info = env.reset()
    for step in range(num_steps):
        action = policy.run_once(obs)
        # action = env.action_space.sample()  # Random action for testing
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 500 == 0:
            logger.info(f"Step {step}: observation keys = {obs.keys() if isinstance(obs, dict) else 'N/A'}")

        if terminated or truncated:
            logger.info(f"Episode ended at step {step}")
            break
    env.close()


@hydra.main(version_base=None, config_path=str(PROJECT_DIR / "configs"), config_name="default")
def main(cfg: DictConfig) -> None:
    """Main function to run Gr00t simulation with Hydra configuration.

    Args:
        cfg: Hydra configuration
    """
    try:
        run(cfg)
    except Exception as e:
        logger.exception(f"An error occurred in main: {e}")
        raise e


if __name__ == "__main__":
    main()

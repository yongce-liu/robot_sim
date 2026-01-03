"""Simple Gr00t simulation example using Hydra configuration."""

from pathlib import Path

import gymnasium as gym
import hydra
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf

# Also triggers task registration with gym.registry
from robot_sim.adapters.gr00t import Gr00tTaskConfig, Gr00tTeleopWrapper  # noqa: F401
from robot_sim.utils.helper import setup_logger
from robot_sim.utils.saver import GymRecorder

PROJECT_DIR = Path(__file__).parents[0].resolve()


def check_observation_space(task: gym.Env, obs: gym.spaces.Dict) -> None:
    for k, v in obs.items():
        space = task.observation_space.spaces[k]
        # logger.info(
        #     f"{k}: value shape={np.shape(v)}, dtype={getattr(v, 'dtype', type(v))}, space shape={space.shape}, space dtype={space.dtype}"
        # )
        if not space.contains(v):
            logger.warning(f"[DEBUG] Observation '{k}' out of space!\nValue: {v}\nSpace: {space}")
            # input("Press Enter to continue...")


def run(cfg: dict) -> None:
    """Run Gr00t simulation.

    Args:
        cfg: Hydra configuration containing simulator_config, observation_mapping, action_mapping
    """
    setup_logger(f"{HydraConfig.get().runtime.output_dir}/{HydraConfig.get().job.name}.loguru.log")
    logger.info("Starting Gr00t simulation...")
    teleop_cfg = cfg.pop("teleop_params", {})
    # Hydra automatically instantiates all _target_ in the config tree
    task_cfg: Gr00tTaskConfig = Gr00tTaskConfig.from_dict(cfg)
    # task_cfg.print()

    # Initialize Gr00tEnv
    logger.info("Initializing Gr00tEnv...")
    _task = gym.make(
        task_cfg.task, config=task_cfg.simulator, maps=task_cfg.maps, **task_cfg.params, render_mode="rgb_array"
    )
    _teleop_wrapper = Gr00tTeleopWrapper(env=_task, **teleop_cfg)
    env = GymRecorder(_teleop_wrapper, include_render=True)

    # Reset environment
    logger.info("Resetting environment...")
    obs, info = env.reset()
    logger.info(f"Initial observation keys: {obs.keys() if isinstance(obs, dict) else 'N/A'}")
    check_observation_space(env, obs)

    while True:
        try:
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action=None)
            # check_observation_space(env, obs)
            if terminated or truncated:
                logger.info("Episode terminated. Resetting environment...")
                obs, info = env.reset()
                logger.info(f"Reset observation keys: {obs.keys() if isinstance(obs, dict) else 'N/A'}")
        except KeyboardInterrupt:
            env.save(output_path=Path(HydraConfig.get().runtime.output_dir) / "recordings", format="npy")
            env.close()
            logger.info("Simulation interrupted by user.")
            break


@hydra.main(version_base=None, config_path=str(PROJECT_DIR / "configs"), config_name="tasks/teleop")
def main(cfg: DictConfig) -> None:
    """Main function to run Gr00t simulation with Hydra configuration.

    Args:
        cfg: Hydra configuration
    """
    try:
        run(OmegaConf.to_container(cfg, resolve=True))
    except Exception as e:
        logger.exception(f"An error occurred in main: {e}")
        raise e


if __name__ == "__main__":
    main()

"""Simple Gr00t simulation example using Hydra configuration."""

from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from robot_sim.adapters.gr00t import Gr00tConfig, Gr00tEnv

PROJECT_ROOT = Path(__file__).parents[2].resolve()


@hydra.main(version_base=None, config_path=str(PROJECT_ROOT / "examples" / "gr00t"), config_name="gr00t")
def main(cfg: DictConfig) -> None:
    """Run Gr00t simulation.

    Args:
        cfg: Hydra configuration containing simulator_config, observation_mapping, action_mapping
    """
    logger.info("Starting Gr00t simulation...")

    # Hydra automatically instantiates all _target_ in the config tree
    gr00t_cfg = Gr00tConfig.from_dict(OmegaConf.to_container(cfg, resolve=True))
    gr00t_cfg.print()

    # Initialize Gr00tEnv
    logger.info("Initializing Gr00tEnv...")
    env = Gr00tEnv(config=gr00t_cfg)

    # logger.info(f"Observation space: {env.observation_space}")
    # logger.info(f"Action space: {env.action_space}")

    # # Reset environment
    # logger.info("Resetting environment...")
    # obs, info = env.reset()
    # logger.info(f"Initial observation keys: {obs.keys() if isinstance(obs, dict) else 'N/A'}")

    # # Run a few simulation steps
    # num_steps = 10
    # logger.info(f"Running {num_steps} simulation steps...")

    # for step in range(num_steps):
    #     # Create a simple random action
    #     # For now, just use zeros for all action groups
    #     action = {}
    #     for action_key in env.action_space.spaces.keys():
    #         action[action_key] = np.zeros(env.action_space.spaces[action_key].shape, dtype=np.float32)

    #     # Step environment
    #     obs, reward, terminated, truncated, info = env.step(action)

    #     if step % 5 == 0:
    #         logger.info(f"Step {step}: observation keys = {obs.keys() if isinstance(obs, dict) else 'N/A'}")

    #     if terminated or truncated:
    #         logger.info(f"Episode ended at step {step}")
    #         break

    # logger.info("Gr00t simulation completed successfully!")
    # env.close()


if __name__ == "__main__":
    main()

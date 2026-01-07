"""Simple Gr00t simulation example using Hydra configuration."""

from pathlib import Path

import gymnasium as gym
import hydra
from hydra.core.hydra_config import HydraConfig
from loguru import logger
from omegaconf import DictConfig, OmegaConf

# Also triggers task registration with gym.registry
from robot_sim.adapters.gr00t import Gr00tDatasetsBuilder, Gr00tTaskConfig, Gr00tTeleopWrapper  # noqa: F401
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


# uv run ./examples/gr00t/run_teleop.py +teleop=default/teleop.yaml
@hydra.main(version_base=None, config_path=str(PROJECT_DIR / "configs"), config_name="tasks/pick_place")
def main(cfg: DictConfig) -> None:
    """Run Gr00t simulation.

    Args:
        cfg: Hydra configuration containing simulator_config, observation_mapping, action_mapping
    """
    setup_logger(f"{HydraConfig.get().runtime.output_dir}/{HydraConfig.get().job.name}.loguru.log")
    logger.info("Starting Gr00t simulation...")
    cfg: dict = OmegaConf.to_container(cfg, resolve=True)
    teleop_path = PROJECT_DIR / f"configs/{cfg.pop('teleop', 'default/teleop.yaml')}"
    teleop_cfg = OmegaConf.to_container(OmegaConf.load(teleop_path), resolve=True)
    # Hydra automatically instantiates all _target_ in the config tree
    task_cfg: Gr00tTaskConfig = Gr00tTaskConfig.from_dict(cfg)
    # task_cfg.print()

    # Initialize Gr00tEnv
    logger.info("Initializing Gr00tEnv...")
    _task = gym.make(
        task_cfg.task, config=task_cfg.simulator, maps=task_cfg.maps, **task_cfg.params, render_mode="rgb_array"
    )
    dataset_builder = Gr00tDatasetsBuilder(env=_task.unwrapped)
    teleop_wrapper = Gr00tTeleopWrapper(env=_task, **teleop_cfg)
    env = GymRecorder(teleop_wrapper, record_reset=False)

    # Reset environment
    logger.info("Resetting environment...")
    obs, info = env.reset()
    logger.info(f"Initial observation keys: {obs.keys() if isinstance(obs, dict) else 'N/A'}")
    check_observation_space(env, obs)

    while True:
        try:
            # Step environment
            action = teleop_wrapper.get_action()
            obs, reward, terminated, truncated, info = env.step(action=action)
            # check_observation_space(env, obs)
            if terminated or truncated:
                logger.info("Episode terminated. Resetting environment...")
                obs, info = env.reset()
                logger.info(f"Reset observation keys: {obs.keys() if isinstance(obs, dict) else 'N/A'}")
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Saving recordings and datasets...")
            env.save(output_path=Path(HydraConfig.get().runtime.output_dir) / "recordings", format="pkl")
            logger.info("Fomatted recordings to gr00t/lerobot-like dataset.")
            dataset_builder.build_dataset(
                source=env._records, output_dir=Path(HydraConfig.get().runtime.output_dir) / "gr00t_datasets"
            )
            env.close()
            logger.info("Simulation interrupted by user.")
            break


if __name__ == "__main__":
    main()

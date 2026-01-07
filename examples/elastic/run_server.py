"""Elastic backend server demo (policy side).

This script runs a Gr00t env with the elastic backend. It waits for states from
the client, computes actions, and sends them back through the bridge.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from robot_sim.adapters.gr00t import Gr00tTaskConfig
from robot_sim.configs import SimulatorConfig

EXAMPLES_DIR = Path(__file__).parents[1].resolve()


def _build_elastic_sim_config(
    base_sim: SimulatorConfig,
    bridge_type: str,
    state_topic: str,
    action_topic: str,
) -> SimulatorConfig:
    sim_dict = deepcopy(base_sim.to_dict())
    sim_dict["backend"] = "elastic"
    sim_dict.setdefault("spec", {})
    sim_dict["spec"]["elastic"] = {
        "bridge": {
            "type": bridge_type,
            "mode": "server",
            "state_topic": state_topic,
            "action_topic": action_topic,
        },
        "state_topic": state_topic,
        "action_topic": action_topic,
        "blocking": True,
    }
    return SimulatorConfig.from_dict(sim_dict)


@hydra.main(version_base=None, config_path=str(EXAMPLES_DIR / "gr00t/configs"), config_name="tasks/pick_place")
def main(cfg: DictConfig) -> None:
    cfg_dict: dict = OmegaConf.to_container(cfg, resolve=True)
    task_cfg = Gr00tTaskConfig.from_dict(cfg_dict)

    elastic_sim = _build_elastic_sim_config(
        task_cfg.simulator,
        bridge_type=cfg_dict.get("bridge_type", "dds"),
        state_topic=cfg_dict.get("state_topic", "rt/robot_sim/state"),
        action_topic=cfg_dict.get("action_topic", "rt/robot_sim/action"),
    )
    elastic_sim.sim.headless = True
    task_cfg.simulator = elastic_sim

    logger.info("Starting elastic server env...")
    env = gym.make(task_cfg.task, config=task_cfg.simulator, maps=task_cfg.maps, **task_cfg.params)

    logger.info("Waiting for client states...")
    obs, info = env.reset()
    logger.info(f"Initial observation keys: {obs.keys() if isinstance(obs, dict) else 'N/A'}")

    while True:
        try:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()
        except KeyboardInterrupt:
            logger.info("Server interrupted.")
            env.close()
            break


if __name__ == "__main__":
    main()

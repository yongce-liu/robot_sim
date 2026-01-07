"""Elastic backend client demo (robot/sim side).

This script runs a local Mujoco backend, receives actions from the server,
applies them, and publishes states back through the bridge.
"""

from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from robot_sim.adapters.gr00t import Gr00tTaskConfig
from robot_sim.backends import BackendFactory, ElasticClientBackend
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
            "mode": "client",
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

    # Local simulator (Mujoco) for producing states.
    sim_cfg = task_cfg.simulator
    sim_cfg.sim.headless = True
    backend = BackendFactory.create_backend(config=sim_cfg, controllers={})
    backend.launch()

    # Elastic client for bridge exchange.
    elastic_sim = _build_elastic_sim_config(
        sim_cfg,
        bridge_type=cfg_dict.get("bridge_type", "dds"),
        state_topic=cfg_dict.get("state_topic", "rt/robot_sim/state"),
        action_topic=cfg_dict.get("action_topic", "rt/robot_sim/action"),
    )
    elastic_backend = ElasticClientBackend(config=elastic_sim)
    elastic_backend.launch()

    logger.info("Publishing initial state...")
    elastic_backend.set_states(backend.get_states())

    try:
        while True:
            actions = elastic_backend.get_actions()
            if actions is None:
                time.sleep(0.001)
                continue
            backend.set_actions(actions)
            backend.simulate()
            elastic_backend.set_states(backend.get_states())
    except KeyboardInterrupt:
        logger.info("Client interrupted.")
    finally:
        elastic_backend.close()
        backend.close()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Script to run simulations with different configurations."""

from pathlib import Path

import hydra
import yaml
from loguru import logger
from omegaconf import DictConfig, OmegaConf

import robot_sim
from robot_sim.backends import BackendFactory
from robot_sim.configs import SimulatorConfig

PROJECT_DIR = Path(robot_sim.__file__).parents[1].resolve()


@hydra.main(version_base=None, config_path=str(PROJECT_DIR / "configs"), config_name="default")
def main(cfg: DictConfig) -> None:
    """Main entry point.

    Switch backends easily:
        python scripts/run_sim.py backend=isaac
        python scripts/run_sim.py backend=mujoco
        python scripts/run_sim.py backend=isaac robot=h1

    Args:
        cfg: Hydra configuration object
    """
    # Print configuration
    cfg = SimulatorConfig.from_dict(OmegaConf.to_container(cfg, resolve=True))
    cfg_dict = OmegaConf.to_container(OmegaConf.create(cfg), resolve=True)
    logger.info("Configuration:\n{}", yaml.dump(cfg_dict))

    # Create simulation manager
    sim_backend = BackendFactory(config=cfg).backend

    # Setup and run simulation
    sim_backend.launch()
    logger.info("Simulation launched successfully.")

    try:
        while True:
            sim_backend.simulate()
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user.")
    finally:
        sim_backend.close()
        logger.info("Simulation closed.")


if __name__ == "__main__":
    main()

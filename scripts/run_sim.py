#!/usr/bin/env python3
"""Script to run simulations with different configurations."""

from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from robot_sim.backends.factory import BackendFactory
from robot_sim.configs import SimulatorConfig

PROJECT_DIR = Path(__file__).resolve().parent.parent


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
    logger.info("Configuration:")
    cfg.print()

    # Create simulation manager
    sim_backend = BackendFactory(config=cfg).backend

    # Setup and run simulation
    sim_backend.launch()
    logger.info("Simulation launched successfully.")

    try:
        while True:
            pass
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user.")
    finally:
        sim_backend.close()
        logger.info("Simulation closed.")


if __name__ == "__main__":
    main()

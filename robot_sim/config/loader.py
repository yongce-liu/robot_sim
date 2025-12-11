"""Configuration loader utilities."""

from pathlib import Path
from typing import Any, Dict

import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf


class ConfigLoader:
    """Load configuration using Hydra."""

    def __init__(self, config_path: Path | str, config_name: str = "default") -> None:
        """Initialize config loader.
        
        Args:
            config_path: Path to configuration directory
            config_name: Name of the configuration file (without .yaml extension)
        """
        self.config_path = Path(config_path).resolve()
        self.config_name = config_name

    def load(self) -> DictConfig:
        """Load configuration using Hydra.
        
        Returns:
            DictConfig: Loaded configuration
        """
        # Clear any existing Hydra instance
        GlobalHydra.instance().clear()
        
        # Initialize Hydra with config directory
        initialize_config_dir(config_dir=str(self.config_path), version_base=None)
        
        # Compose configuration
        cfg = compose(config_name=self.config_name)
        
        return cfg

    def save(self, config: DictConfig, output_path: Path | str) -> None:
        """Save configuration to YAML file.
        
        Args:
            config: Configuration to save
            output_path: Path to save the configuration
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            f.write(OmegaConf.to_yaml(config))

    @staticmethod
    def merge_configs(*configs: DictConfig) -> DictConfig:
        """Merge multiple configurations.
        
        Args:
            *configs: Configurations to merge
            
        Returns:
            DictConfig: Merged configuration
        """
        return OmegaConf.merge(*configs)

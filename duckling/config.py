"""Configuration management module.

This module provides the Config class for loading and accessing configuration
data from YAML files with support for nested key access.
"""

from typing import Any
from pathlib import Path
import yaml


class Config:
    """Load and manage application configuration from YAML file.

    Provides access to configuration sections (models, prompts) and individual
    values with optional key-based access.
    """

    def __init__(self, path: str | None = None) -> None:
        """Initialize the Config class by loading config.yaml.

        Args:
            path: Optional path to the configuration file. If not provided,
                        defaults to config.yaml in the module directory.

        Raises:
            FileNotFoundError: If the configuration file is not found.
        """
        if path is None:
            config_path = Path(__file__).parent / "config.yaml"
        else:
            config_path = Path(path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            self.data = yaml.safe_load(f)

    def models(self, key: str) -> Any:
        """Get specific model configuration.

        Args:
            key: Specific model key to retrieve.

        Returns:
            The value for the specified model key.

        Raises:
            KeyError: If the key is not found in models configuration.
        """
        models = self.data.get("models", {})
        if key not in models:
            raise KeyError(f"Model key '{key}' not found in configuration")
        return models[key]

    def prompts(self, key: str) -> Any:
        """Get specific prompt configuration.

        Args:
            key: Specific prompt key to retrieve.

        Returns:
            The value for the specified prompt key.

        Raises:
            KeyError: If the key is not found in prompts configuration.
        """
        prompts = self.data.get("prompts", {})
        if key not in prompts:
            raise KeyError(f"Prompt key '{key}' not found in configuration")
        return prompts[key]

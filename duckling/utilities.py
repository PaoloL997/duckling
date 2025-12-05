"""Utility functions for logging, device detection, and configuration management."""

from typing import Any, Optional, Union
import logging
import os
from pathlib import Path
import torch
import yaml


def setup_logger(
    name: str = __name__, log_file: str = "logs/document_processor.log"
) -> logging.Logger:
    """Setup logging configuration and return logger instance.

    Creates a logger with both console and file handlers. Ensures the logs
    directory exists before attempting to write log files.

    Args:
        name: Logger name, typically __name__. Defaults to __name__.
        log_file: Path to the log file. Defaults to "logs/document_processor.log".

    Returns:
        logging.Logger: Configured logger instance.
    """
    os.makedirs("logs", exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode="a", encoding="utf-8"),
        ],
    )
    return logging.getLogger(name)


def check_device() -> str:
    """Detect and return the available device for computation.

    Checks for GPU availability (CUDA), then MPS (Metal Performance Shaders),
    and defaults to CPU if neither is available.

    Returns:
        str: Device type - "cuda", "mps", or "cpu".
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device


class ConfigManager:
    """Loads and accesses YAML configuration files.

    Provides a simple interface to load and query YAML configuration files
    with support for nested key access using dot notation.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize the ConfigManager.

        Args:
            config_path: Path to the YAML config file. If None, looks for
                        config.yaml in the same directory as this file.
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """Get a configuration value using dot notation.

        Args:
            key: Dot-separated key path (e.g., "models.embedding").
            default: Default value if key not found. Defaults to None.

        Returns:
            The configuration value or default if not found.
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            value = value.get(k) if isinstance(value, dict) else None
            if value is None:
                return default

        return value

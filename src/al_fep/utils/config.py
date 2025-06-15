"""
Configuration utilities
"""

import yaml
import os
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str, default_config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to main config file
        default_config_path: Path to default config file
        
    Returns:
        Configuration dictionary
    """
    config = {}
    
    # Load default config first
    if default_config_path and os.path.exists(default_config_path):
        try:
            with open(default_config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded default config from {default_config_path}")
        except Exception as e:
            logger.warning(f"Failed to load default config: {e}")
    
    # Override with specific config
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                specific_config = yaml.safe_load(f)
                config.update(specific_config)
            logger.info(f"Loaded config from {config_path}")
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            raise
    else:
        logger.warning(f"Config file not found: {config_path}")
    
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Output file path
    """
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Config saved to {config_path}")
        
    except Exception as e:
        logger.error(f"Failed to save config: {e}")
        raise


def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge multiple configuration dictionaries.
    
    Args:
        *configs: Configuration dictionaries to merge
        
    Returns:
        Merged configuration
    """
    merged = {}
    
    for config in configs:
        if config:
            merged.update(config)
    
    return merged

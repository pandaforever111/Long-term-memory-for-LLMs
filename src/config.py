#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Configuration Module

This module handles configuration loading and management for the GPT Memory Agent.
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import yaml


class Config:
    """Configuration manager for the GPT Memory Agent."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration from file or defaults.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
        """
        # Set up default configuration
        self._set_defaults()
        
        # Load configuration from file if provided
        if config_path:
            self._load_from_file(config_path)
            
        # Override with environment variables
        self._load_from_env()

    def _set_defaults(self) -> None:
        """Set default configuration values."""
        # General settings
        self.app_name = "GPT Memory Agent"
        self.version = "0.1.0"
        self.log_level = logging.INFO
        
        # Paths and storage
        self.data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
        self.db_path = os.path.join(self.data_dir, "memories.db")
        
        # AI Provider setting
        self.ai_provider = "openai"  # Options: "openai", "puter"
        
        # OpenAI API settings
        self.openai_api_key = ""
        self.model_name = "gpt-4o"
        self.embedding_model = "text-embedding-3-small"
        self.temperature = 0.7
        self.max_tokens = 1000
        self.top_p = 1.0
        self.frequency_penalty = 0.0
        self.presence_penalty = 0.0
        
        # Memory settings
        self.max_memories_per_user = 1000
        self.memory_retention_days = 365  # How long to keep memories by default
        self.memory_importance_threshold = 0.3  # Minimum importance to store a memory
        
        # Text processing settings
        self.use_spacy = True
        self.use_nltk = True
        self.spacy_model = "en_core_web_sm"
        
        # System prompt
        self.system_prompt = """
        You are an AI assistant with long-term memory capabilities. You can remember 
        information about the user from previous conversations and use it to provide 
        more personalized and contextually relevant responses.
        
        When the user shares personal information, preferences, or important facts, 
        you should remember these details and reference them appropriately in future 
        interactions. However, be mindful of privacy and only use remembered information 
        when it's relevant to the current conversation.
        
        If the user asks you to forget certain information, acknowledge their request 
        and confirm that you will no longer reference that information.
        
        Always be helpful, accurate, and respectful in your responses.
        """.strip()

    def _load_from_file(self, config_path: str) -> None:
        """Load configuration from a file.
        
        Args:
            config_path: Path to configuration file (YAML or JSON)
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        file_ext = os.path.splitext(config_path)[1].lower()
        
        try:
            with open(config_path, 'r') as f:
                if file_ext == '.json':
                    config_data = json.load(f)
                elif file_ext in ['.yaml', '.yml']:
                    config_data = yaml.safe_load(f)
                else:
                    raise ValueError(f"Unsupported configuration file format: {file_ext}")
                    
            # Update configuration with loaded values
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                    
        except Exception as e:
            raise ValueError(f"Error loading configuration from {config_path}: {e}")

    def _load_from_env(self) -> None:
        """Override configuration with environment variables."""
        # Map of environment variable names to config attributes
        env_mapping = {
            "MEMORY_AGENT_AI_PROVIDER": "ai_provider",
            "OPENAI_API_KEY": "openai_api_key",
            "MEMORY_AGENT_MODEL": "model_name",
            "MEMORY_AGENT_EMBEDDING_MODEL": "embedding_model",
            "MEMORY_AGENT_TEMPERATURE": "temperature",
            "MEMORY_AGENT_MAX_TOKENS": "max_tokens",
            "MEMORY_AGENT_DB_PATH": "db_path",
            "MEMORY_AGENT_LOG_LEVEL": "log_level"
        }
        
        for env_var, config_attr in env_mapping.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                # Convert value to appropriate type based on default
                default_value = getattr(self, config_attr)
                if isinstance(default_value, bool):
                    env_value = env_value.lower() in ['true', '1', 'yes', 'y']
                elif isinstance(default_value, int):
                    env_value = int(env_value)
                elif isinstance(default_value, float):
                    env_value = float(env_value)
                elif config_attr == "log_level" and isinstance(env_value, str):
                    # Convert string log level to logging constant
                    log_levels = {
                        "debug": logging.DEBUG,
                        "info": logging.INFO,
                        "warning": logging.WARNING,
                        "error": logging.ERROR,
                        "critical": logging.CRITICAL
                    }
                    env_value = log_levels.get(env_value.lower(), logging.INFO)
                    
                # Update the configuration
                setattr(self, config_attr, env_value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary of configuration values
        """
        return {key: value for key, value in self.__dict__.items() 
                if not key.startswith('_') and not callable(value)}

    def save_to_file(self, file_path: str) -> None:
        """Save current configuration to a file.
        
        Args:
            file_path: Path to save configuration file
        """
        config_dict = self.to_dict()
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            with open(file_path, 'w') as f:
                if file_ext == '.json':
                    json.dump(config_dict, f, indent=2)
                elif file_ext in ['.yaml', '.yml']:
                    yaml.dump(config_dict, f, default_flow_style=False)
                else:
                    raise ValueError(f"Unsupported configuration file format: {file_ext}")
                    
        except Exception as e:
            raise ValueError(f"Error saving configuration to {file_path}: {e}")


def create_default_config(config_path: str) -> None:
    """Create a default configuration file.
    
    Args:
        config_path: Path to save the default configuration
    """
    config = Config()
    config.save_to_file(config_path)
    print(f"Default configuration saved to {config_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GPT Memory Agent Configuration")
    parser.add_argument(
        "--create-config", 
        type=str,
        help="Create a default configuration file at the specified path"
    )
    args = parser.parse_args()
    
    if args.create_config:
        create_default_config(args.create_config)
    else:
        # Print the default configuration
        config = Config()
        print(json.dumps(config.to_dict(), indent=2))
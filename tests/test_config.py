#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for the Config class.
"""

import unittest
import os
import sys
import tempfile
import yaml
import json
from unittest.mock import patch

# Add the parent directory to the path so we can import the src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import Config


class TestConfig(unittest.TestCase):
    """Test cases for the Config class."""

    def setUp(self):
        """Set up temporary files for testing."""
        # Create a temporary directory for test files
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a test YAML config file
        self.yaml_config_path = os.path.join(self.temp_dir.name, "test_config.yaml")
        yaml_config = {
            "app": {
                "name": "Test App",
                "version": "1.0.0",
                "log_level": "INFO"
            },
            "paths": {
                "data_dir": "/path/to/data",
                "log_dir": "/path/to/logs"
            },
            "openai": {
                "api_key": "test_api_key",
                "model": "gpt-4o",
                "embedding_model": "text-embedding-3-small",
                "temperature": 0.7,
                "max_tokens": 1000
            },
            "memory": {
                "max_memories_per_user": 1000,
                "retention_days": 30,
                "importance_threshold": 0.5
            },
            "text_processing": {
                "spacy_model": "en_core_web_sm",
                "nltk_data_path": "/path/to/nltk_data"
            },
            "system_prompt": "You are a helpful assistant with memory capabilities."
        }
        
        with open(self.yaml_config_path, "w") as f:
            yaml.dump(yaml_config, f)
        
        # Create a test JSON config file
        self.json_config_path = os.path.join(self.temp_dir.name, "test_config.json")
        with open(self.json_config_path, "w") as f:
            json.dump(yaml_config, f)

    def tearDown(self):
        """Clean up temporary files after testing."""
        self.temp_dir.cleanup()

    def test_default_initialization(self):
        """Test initializing with default values."""
        config = Config()
        
        # Check that default values were set
        self.assertEqual(config.app["name"], "GPT Memory Agent")
        self.assertEqual(config.app["version"], "0.1.0")
        self.assertEqual(config.app["log_level"], "INFO")
        
        self.assertIsNotNone(config.paths["data_dir"])
        self.assertIsNotNone(config.paths["log_dir"])
        
        self.assertIsNone(config.openai["api_key"])
        self.assertEqual(config.openai["model"], "gpt-4o")
        self.assertEqual(config.openai["embedding_model"], "text-embedding-3-small")
        self.assertEqual(config.openai["temperature"], 0.7)
        self.assertEqual(config.openai["max_tokens"], 1000)
        
        self.assertEqual(config.memory["max_memories_per_user"], 1000)
        self.assertEqual(config.memory["retention_days"], 30)
        self.assertEqual(config.memory["importance_threshold"], 0.5)
        
        self.assertEqual(config.text_processing["spacy_model"], "en_core_web_sm")
        self.assertIsNone(config.text_processing["nltk_data_path"])
        
        self.assertIsNotNone(config.system_prompt)

    def test_load_from_yaml(self):
        """Test loading configuration from a YAML file."""
        config = Config()
        config.load_from_file(self.yaml_config_path)
        
        # Check that values were loaded from the YAML file
        self.assertEqual(config.app["name"], "Test App")
        self.assertEqual(config.app["version"], "1.0.0")
        self.assertEqual(config.app["log_level"], "INFO")
        
        self.assertEqual(config.paths["data_dir"], "/path/to/data")
        self.assertEqual(config.paths["log_dir"], "/path/to/logs")
        
        self.assertEqual(config.openai["api_key"], "test_api_key")
        self.assertEqual(config.openai["model"], "gpt-4o")
        self.assertEqual(config.openai["embedding_model"], "text-embedding-3-small")
        self.assertEqual(config.openai["temperature"], 0.7)
        self.assertEqual(config.openai["max_tokens"], 1000)
        
        self.assertEqual(config.memory["max_memories_per_user"], 1000)
        self.assertEqual(config.memory["retention_days"], 30)
        self.assertEqual(config.memory["importance_threshold"], 0.5)
        
        self.assertEqual(config.text_processing["spacy_model"], "en_core_web_sm")
        self.assertEqual(config.text_processing["nltk_data_path"], "/path/to/nltk_data")
        
        self.assertEqual(config.system_prompt, "You are a helpful assistant with memory capabilities.")

    def test_load_from_json(self):
        """Test loading configuration from a JSON file."""
        config = Config()
        config.load_from_file(self.json_config_path)
        
        # Check that values were loaded from the JSON file
        self.assertEqual(config.app["name"], "Test App")
        self.assertEqual(config.openai["model"], "gpt-4o")
        self.assertEqual(config.memory["max_memories_per_user"], 1000)
        self.assertEqual(config.text_processing["spacy_model"], "en_core_web_sm")
        self.assertEqual(config.system_prompt, "You are a helpful assistant with memory capabilities.")

    def test_load_nonexistent_file(self):
        """Test loading from a nonexistent file."""
        config = Config()
        
        # Attempt to load from a nonexistent file
        with self.assertRaises(FileNotFoundError):
            config.load_from_file("/path/to/nonexistent/file.yaml")

    def test_load_invalid_file_format(self):
        """Test loading from a file with an invalid format."""
        # Create a file with invalid format
        invalid_file_path = os.path.join(self.temp_dir.name, "invalid.txt")
        with open(invalid_file_path, "w") as f:
            f.write("This is not a valid YAML or JSON file.")
        
        config = Config()
        
        # Attempt to load from a file with invalid format
        with self.assertRaises(ValueError):
            config.load_from_file(invalid_file_path)

    @patch.dict(os.environ, {
        "MEMORY_AGENT_OPENAI_API_KEY": "env_api_key",
        "MEMORY_AGENT_OPENAI_MODEL": "gpt-3.5-turbo",
        "MEMORY_AGENT_MEMORY_MAX_MEMORIES_PER_USER": "500",
        "MEMORY_AGENT_APP_LOG_LEVEL": "DEBUG"
    })
    def test_load_from_env(self):
        """Test loading configuration from environment variables."""
        config = Config()
        config.load_from_env()
        
        # Check that values were loaded from environment variables
        self.assertEqual(config.openai["api_key"], "env_api_key")
        self.assertEqual(config.openai["model"], "gpt-3.5-turbo")
        self.assertEqual(config.memory["max_memories_per_user"], 500)
        self.assertEqual(config.app["log_level"], "DEBUG")

    def test_save_to_file(self):
        """Test saving configuration to a file."""
        config = Config()
        config.app["name"] = "Saved Config"
        config.openai["api_key"] = "saved_api_key"
        
        # Save to a YAML file
        save_path = os.path.join(self.temp_dir.name, "saved_config.yaml")
        config.save_to_file(save_path)
        
        # Load the saved file and check the values
        loaded_config = Config()
        loaded_config.load_from_file(save_path)
        
        self.assertEqual(loaded_config.app["name"], "Saved Config")
        self.assertEqual(loaded_config.openai["api_key"], "saved_api_key")

    def test_get_item(self):
        """Test getting configuration items using dictionary-like access."""
        config = Config()
        config.app["name"] = "Test App"
        config.openai["api_key"] = "test_api_key"
        
        # Test getting items
        self.assertEqual(config["app.name"], "Test App")
        self.assertEqual(config["openai.api_key"], "test_api_key")
        
        # Test getting nonexistent items
        with self.assertRaises(KeyError):
            _ = config["nonexistent.key"]

    def test_set_item(self):
        """Test setting configuration items using dictionary-like access."""
        config = Config()
        
        # Test setting items
        config["app.name"] = "New App Name"
        config["openai.api_key"] = "new_api_key"
        
        self.assertEqual(config.app["name"], "New App Name")
        self.assertEqual(config.openai["api_key"], "new_api_key")
        
        # Test setting nonexistent items
        with self.assertRaises(KeyError):
            config["nonexistent.key"] = "value"

    def test_to_dict(self):
        """Test converting configuration to a dictionary."""
        config = Config()
        config.app["name"] = "Test App"
        config.openai["api_key"] = "test_api_key"
        
        # Convert to dictionary
        config_dict = config.to_dict()
        
        # Check the dictionary
        self.assertEqual(config_dict["app"]["name"], "Test App")
        self.assertEqual(config_dict["openai"]["api_key"], "test_api_key")


if __name__ == "__main__":
    unittest.main()
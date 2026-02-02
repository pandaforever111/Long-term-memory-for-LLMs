#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for the AIClientFactory class.
"""

import unittest
import os
import sys
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ai_client_factory import AIClientFactory
from src.config import Config


class TestAIClientFactory(unittest.TestCase):
    """Test cases for the AIClientFactory class."""

    def setUp(self):
        """Set up a Config instance for testing."""
        # Create a minimal configuration
        self.config = Config()
        self.config.openai = {
            "api_key": "test_api_key",
            "model": "gpt-4o",
            "embedding_model": "text-embedding-3-small",
            "temperature": 0.7,
            "max_tokens": 1000,
            "system_prompt": "You are a helpful assistant with memory capabilities."
        }

    @patch('src.ai_client_factory.OpenAIClient')
    def test_create_openai_client(self, mock_openai_client):
        """Test creating an OpenAI client."""
        # Set the AI provider to OpenAI
        self.config.ai_provider = "openai"
        
        # Create a client using the factory
        client = AIClientFactory.create_client(self.config)
        
        # Check that the OpenAIClient was initialized with the correct config
        mock_openai_client.assert_called_once_with(self.config)
        
        # Check that the factory returned the mocked OpenAIClient
        self.assertEqual(client, mock_openai_client.return_value)

    @patch('src.ai_client_factory.PuterClient')
    def test_create_puter_client(self, mock_puter_client):
        """Test creating a Puter.js client."""
        # Set the AI provider to Puter
        self.config.ai_provider = "puter"
        
        # Create a client using the factory
        client = AIClientFactory.create_client(self.config)
        
        # Check that the PuterClient was initialized with the correct config
        mock_puter_client.assert_called_once_with(self.config)
        
        # Check that the factory returned the mocked PuterClient
        self.assertEqual(client, mock_puter_client.return_value)

    @patch('src.ai_client_factory.PuterClient')
    def test_create_puter_js_client(self, mock_puter_client):
        """Test creating a Puter.js client with 'puter.js' provider name."""
        # Set the AI provider to Puter.js
        self.config.ai_provider = "puter.js"
        
        # Create a client using the factory
        client = AIClientFactory.create_client(self.config)
        
        # Check that the PuterClient was initialized with the correct config
        mock_puter_client.assert_called_once_with(self.config)
        
        # Check that the factory returned the mocked PuterClient
        self.assertEqual(client, mock_puter_client.return_value)

    def test_create_unsupported_client(self):
        """Test creating a client with an unsupported provider."""
        # Set the AI provider to an unsupported value
        self.config.ai_provider = "unsupported_provider"
        
        # Check that creating a client raises a ValueError
        with self.assertRaises(ValueError):
            AIClientFactory.create_client(self.config)

    @patch('src.ai_client_factory.OPENAI_AVAILABLE', False)
    def test_openai_not_available(self):
        """Test creating an OpenAI client when the OpenAI package is not available."""
        # Set the AI provider to OpenAI
        self.config.ai_provider = "openai"
        
        # Check that creating a client raises an ImportError
        with self.assertRaises(ImportError):
            AIClientFactory.create_client(self.config)

    @patch('src.ai_client_factory.PUTER_AVAILABLE', False)
    def test_puter_not_available(self):
        """Test creating a Puter.js client when the Puter.js client is not available."""
        # Set the AI provider to Puter
        self.config.ai_provider = "puter"
        
        # Check that creating a client raises an ImportError
        with self.assertRaises(ImportError):
            AIClientFactory.create_client(self.config)


if __name__ == "__main__":
    unittest.main()
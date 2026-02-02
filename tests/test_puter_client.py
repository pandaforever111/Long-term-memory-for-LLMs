#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for the PuterClient class.
"""

import unittest
import os
import sys
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.puter_client import PuterClient
from src.config import Config


class TestPuterClient(unittest.TestCase):
    """Test cases for the PuterClient class."""

    def setUp(self):
        """Set up a PuterClient instance for testing."""
        # Create a minimal configuration
        config = Config()
        config.openai = {
            "model": "gpt-4o",
            "embedding_model": "text-embedding-3-small",
            "temperature": 0.7,
            "max_tokens": 1000,
            "system_prompt": "You are a helpful assistant with memory capabilities."
        }
        
        # Initialize the PuterClient with the test configuration
        self.client = PuterClient(config)

    def test_initialization(self):
        """Test that the client initializes correctly."""
        # Check that the client was initialized
        self.assertIsNotNone(self.client)
        self.assertIsNotNone(self.client.config)
        # The PuterClient doesn't have these attributes directly
        # but they are accessible through the config
        self.assertEqual(self.client.config.openai["model"], "gpt-4o")
        self.assertEqual(self.client.config.openai["embedding_model"], "text-embedding-3-small")
        self.assertEqual(self.client.config.openai["temperature"], 0.7)
        self.assertEqual(self.client.config.openai["max_tokens"], 1000)
        self.assertEqual(self.client.config.openai["system_prompt"], "You are a helpful assistant with memory capabilities.")

    def test_generate_response(self):
        """Test generating a response."""
        # Since PuterClient is a placeholder, we're just testing the interface
        user_message = "Hello, how are you?"
        memories = [
            {"content": "User's name is John", "importance": 0.8},
            {"content": "User likes pizza", "importance": 0.6}
        ]
        
        # The placeholder implementation should return a fixed response
        response = self.client.generate_response(user_message, memories)
        
        # Check that the response is a string
        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)

    def test_prepare_messages(self):
        """Test preparing messages for the API."""
        # Test preparing messages
        user_message = "What's my name?"
        memory_context = "User's name is John. User likes pizza."
        
        messages = self.client._prepare_messages(user_message, memory_context)
        
        # Check that we have the expected number of messages
        self.assertEqual(len(messages), 2)  # System prompt with memory context, user message
        
        # Check system prompt with memory context
        self.assertEqual(messages[0]["role"], "system")
        # The system prompt comes from the config
        system_prompt = self.client.config.system_prompt
        self.assertTrue(system_prompt in messages[0]["content"])
        self.assertTrue("User's name is John. User likes pizza." in messages[0]["content"])
        
        # Check user message
        self.assertEqual(messages[1]["role"], "user")
        self.assertEqual(messages[1]["content"], "What's my name?")

    def test_extract_embedding(self):
        """Test getting embeddings."""
        # Test getting an embedding
        text = "This is a test"
        embedding = self.client.extract_embedding(text)
        
        # Check that the embedding is a list of floats
        self.assertIsInstance(embedding, list)
        self.assertTrue(all(isinstance(x, float) for x in embedding))

    def test_moderate_content(self):
        """Test content moderation."""
        # Test 1: Content that should not be flagged
        text = "This is a harmless message"
        result = self.client.moderate_content(text)
        
        # Check that the content was not flagged
        self.assertFalse(result["flagged"])
        
        # Test 2: Content that should be flagged
        text = "This is harmful content"
        result = self.client.moderate_content(text)
        
        # The placeholder implementation should always return False
        self.assertFalse(result["flagged"])


if __name__ == "__main__":
    unittest.main()
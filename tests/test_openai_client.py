#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for the OpenAIClient class.
"""

import unittest
import os
import sys
from unittest.mock import patch, MagicMock

# Add the parent directory to the path so we can import the src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.openai_client import OpenAIClient
from src.config import Config


class TestOpenAIClient(unittest.TestCase):
    """Test cases for the OpenAIClient class."""

    def setUp(self):
        """Set up an OpenAIClient instance for testing with mocked OpenAI API."""
        # Create a minimal configuration
        config = Config()
        config.openai = {
            "api_key": "test_api_key",
            "model": "gpt-4o",
            "embedding_model": "text-embedding-3-small",
            "temperature": 0.7,
            "max_tokens": 1000,
            "system_prompt": "You are a helpful assistant with memory capabilities."
        }
        
        # Patch the OpenAI client initialization
        patcher = patch('openai.OpenAI')
        self.mock_openai = patcher.start()
        self.addCleanup(patcher.stop)
        
        # Initialize the OpenAIClient with the test configuration
        self.client = OpenAIClient(config)

    def test_initialization(self):
        """Test that the client initializes correctly."""
        # Check that the OpenAI client was initialized with the correct API key
        self.mock_openai.assert_called_once_with(api_key="test_api_key")
        
        # Check that the configuration was set correctly
        self.assertEqual(self.client.model, "gpt-4o")
        self.assertEqual(self.client.embedding_model, "text-embedding-3-small")
        self.assertEqual(self.client.temperature, 0.7)
        self.assertEqual(self.client.max_tokens, 1000)
        self.assertEqual(self.client.system_prompt, "You are a helpful assistant with memory capabilities.")

    def test_generate_response(self):
        """Test generating a response from the OpenAI API."""
        # Create a mock response from the OpenAI API
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message = MagicMock()
        mock_completion.choices[0].message.content = "This is a test response."
        
        # Set up the mock to return our mock completion
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_completion
        self.mock_openai.return_value = mock_client
        
        # Test generating a response
        user_message = "Hello, how are you?"
        memories = [
            {"content": "User's name is John", "importance": 0.8},
            {"content": "User likes pizza", "importance": 0.6}
        ]
        
        response = self.client.generate_response(user_message, memories)
        
        # Check that the response matches our mock
        self.assertEqual(response, "This is a test response.")
        
        # Check that the API was called with the correct parameters
        mock_client.chat.completions.create.assert_called_once()
        call_args = mock_client.chat.completions.create.call_args[1]
        
        # Check model and parameters
        self.assertEqual(call_args["model"], "gpt-4o")
        self.assertEqual(call_args["temperature"], 0.7)
        self.assertEqual(call_args["max_tokens"], 1000)
        
        # Check messages
        messages = call_args["messages"]
        
        # First message should be the system prompt
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "You are a helpful assistant with memory capabilities.")
        
        # There should be a message containing memory context
        memory_message_found = False
        for message in messages:
            if message["role"] == "system" and "User's name is John" in message["content"] and "User likes pizza" in message["content"]:
                memory_message_found = True
                break
        self.assertTrue(memory_message_found, "Memory context not found in messages")
        
        # Last message should be the user's message
        self.assertEqual(messages[-1]["role"], "user")
        self.assertEqual(messages[-1]["content"], "Hello, how are you?")

    def test_prepare_messages(self):
        """Test preparing messages for the OpenAI API."""
        # Test preparing messages
        user_message = "What's my name?"
        memories = [
            {"content": "User's name is John", "importance": 0.8},
            {"content": "User likes pizza", "importance": 0.6}
        ]
        
        messages = self.client.prepare_messages(user_message, memories)
        
        # Check that we have the expected number of messages
        self.assertEqual(len(messages), 3)  # System prompt, memory context, user message
        
        # Check system prompt
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[0]["content"], "You are a helpful assistant with memory capabilities.")
        
        # Check memory context
        self.assertEqual(messages[1]["role"], "system")
        self.assertTrue("User's name is John" in messages[1]["content"])
        self.assertTrue("User likes pizza" in messages[1]["content"])
        
        # Check user message
        self.assertEqual(messages[2]["role"], "user")
        self.assertEqual(messages[2]["content"], "What's my name?")

    def test_get_embedding(self):
        """Test getting embeddings from the OpenAI API."""
        # Create a mock response from the OpenAI API
        mock_embedding_response = MagicMock()
        mock_embedding_response.data = [MagicMock()]
        mock_embedding_response.data[0].embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        # Set up the mock to return our mock embedding response
        self.mock_client.embeddings.create.return_value = mock_embedding_response
        
        # Test getting an embedding
        text = "This is a test"
        embedding = self.client.get_embedding(text)
        
        # Check that the embedding matches our mock
        self.assertEqual(embedding, [0.1, 0.2, 0.3, 0.4, 0.5])
        
        # Check that the API was called with the correct parameters
        self.mock_client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=text
        )

    def test_moderate_content(self):
        """Test content moderation with the OpenAI API."""
        # Create a mock response from the OpenAI API
        mock_moderation_response = MagicMock()
        mock_moderation_response.results = [MagicMock()]
        
        # Set up the mock to return different flagged values for different tests
        self.mock_client.moderations.create.return_value = mock_moderation_response
        
        # Test 1: Content that should not be flagged
        mock_moderation_response.results[0].flagged = False
        
        text = "This is a harmless message"
        is_flagged = self.client.moderate_content(text)
        
        # Check that the content was not flagged
        self.assertFalse(is_flagged)
        
        # Check that the API was called with the correct parameters
        self.mock_client.moderations.create.assert_called_with(input=text)
        
        # Test 2: Content that should be flagged
        mock_moderation_response.results[0].flagged = True
        
        text = "This is harmful content"
        is_flagged = self.client.moderate_content(text)
        
        # Check that the content was flagged
        self.assertTrue(is_flagged)
        
        # Check that the API was called with the correct parameters
        self.mock_client.moderations.create.assert_called_with(input=text)


if __name__ == "__main__":
    unittest.main()
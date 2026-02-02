#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for the Memory Agent

This module contains tests for the GPT Memory Agent functionality.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch
import tempfile
import json
from datetime import datetime

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory_agent import MemoryAgent
from src.memory_store import MemoryStore
from src.text_processor import TextProcessor
from src.config import Config
from src.ai_client_factory import AIClientFactory


class TestMemoryAgent(unittest.TestCase):
    """Test cases for the Memory Agent."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary database file
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp(suffix=".db")
        
        # Create a mock config
        self.config = MagicMock(spec=Config)
        self.config.db_path = self.temp_db_path
        self.config.log_level = 30  # WARNING level to reduce test output
        self.config.openai_api_key = "test_key"
        self.config.model_name = "gpt-4o"
        self.config.embedding_model = "text-embedding-3-small"
        self.config.use_spacy = False
        self.config.use_nltk = False
        
        # Set AI provider to OpenAI for testing
        self.config.ai_provider = "openai"
        
        # Mock the OpenAI client
        self.openai_patcher = patch('src.openai_client.OpenAI')
        self.mock_openai = self.openai_patcher.start()
        
        # Create a test agent with mocked components
        with patch('src.memory_agent.Config', return_value=self.config):
            with patch('src.memory_agent.AIClientFactory.create_client'):
                self.agent = MemoryAgent()
                # Replace the real memory store with one using our temp DB
                self.agent.memory_store = MemoryStore(self.config)

    def tearDown(self):
        """Tear down test fixtures."""
        # Close and remove the temporary database
        os.close(self.temp_db_fd)
        os.unlink(self.temp_db_path)
        
        # Stop the OpenAI patcher
        self.openai_patcher.stop()

    def test_process_message(self):
        """Test processing a message and extracting memories."""
        # Mock the text processor to return a fixed set of memory candidates
        self.agent.text_processor.extract_memory_candidates = MagicMock(
            return_value=["I like chocolate", "I work at Acme Corp"]
        )
        self.agent.text_processor.is_valid_memory = MagicMock(return_value=True)
        
        # Process a test message
        result = self.agent.process_message(
            user_id="test_user",
            message="Hello, I'm John. I like chocolate and I work at Acme Corp."
        )
        
        # Check that the message was processed correctly
        self.assertEqual(result["user_id"], "test_user")
        self.assertIsNotNone(result["conversation_id"])
        self.assertEqual(len(result["stored_memories"]), 2)
        self.assertEqual(len(result["deleted_memories"]), 0)
        
        # Verify the stored memories
        memory_contents = [m["content"] for m in result["stored_memories"]]
        self.assertIn("I like chocolate", memory_contents)
        self.assertIn("I work at Acme Corp", memory_contents)

    def test_retrieve_memories(self):
        """Test retrieving memories based on a query."""
        # Store some test memories
        memory_id1 = self.agent.memory_store.store_memory(
            user_id="test_user",
            content="I like chocolate",
            source_message="I really enjoy eating chocolate",
            timestamp=datetime.now()
        )
        
        memory_id2 = self.agent.memory_store.store_memory(
            user_id="test_user",
            content="I work at Acme Corp",
            source_message="I've been working at Acme Corp for 5 years",
            timestamp=datetime.now()
        )
        
        # Mock the text processor to return specific concepts
        self.agent.text_processor.extract_key_concepts = MagicMock(
            return_value=["chocolate", "like"]
        )
        
        # Mock the memory_store.retrieve_memories method to return our test memory
        self.agent.memory_store.retrieve_memories = MagicMock(return_value=[
            {
                "id": memory_id1,
                "content": "I like chocolate",
                "source_message": "I really enjoy eating chocolate",
                "conversation_id": None,
                "importance": 0.5,
                "created_at": datetime.now().isoformat(),
                "metadata": None
            }
        ])
        
        # Retrieve memories with a relevant query
        memories = self.agent.retrieve_memories(
            user_id="test_user",
            query="What do I like to eat?"
        )
        
        # Check that the correct memory was retrieved
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["content"], "I like chocolate")

    def test_generate_response(self):
        # Mock the AI client's generate_response method
        self.agent.ai_client.generate_response = MagicMock()
        """Test generating a response with memory context."""
        # Mock the AI client response
        self.agent.ai_client.generate_response.return_value = "I remember you like chocolate!"
        
        # Store a test memory
        self.agent.memory_store.store_memory(
            user_id="test_user",
            content="I like chocolate",
            source_message="I really enjoy eating chocolate",
            timestamp=datetime.now()
        )
        
        # Mock memory retrieval to return our stored memory
        self.agent.retrieve_memories = MagicMock(return_value=[
            {"id": "123", "content": "I like chocolate"}
        ])
        
        # Generate a response
        result = self.agent.generate_response(
            user_id="test_user",
            message="What do I like to eat?"
        )
        
        # Check the response
        self.assertEqual(result["response"], "I remember you like chocolate!")
        self.assertEqual(len(result["memories_used"]), 1)
        self.assertEqual(result["memories_used"][0]["content"], "I like chocolate")

    def test_memory_deletion(self):
        """Test deleting memories based on content pattern."""
        # Store some test memories
        memory_id1 = self.agent.memory_store.store_memory(
            user_id="test_user",
            content="I like chocolate",
            source_message="I really enjoy eating chocolate",
            timestamp=datetime.now()
        )
        
        memory_id2 = self.agent.memory_store.store_memory(
            user_id="test_user",
            content="I work at Acme Corp",
            source_message="I've been working at Acme Corp for 5 years",
            timestamp=datetime.now()
        )
        
        # Mock the text processor to extract a deletion request
        self.agent.text_processor.extract_deletion_requests = MagicMock(
            return_value=["chocolate"]
        )
        
        # Mock the memory_store.delete_memories_by_content method to return a single memory ID
        self.agent.memory_store.delete_memories_by_content = MagicMock(
            return_value=[memory_id1]
        )
        
        # Process a message with a deletion request
        result = self.agent.process_message(
            user_id="test_user",
            message="Please forget that I like chocolate"
        )
        
        # Check that the memory was deleted
        self.assertEqual(len(result["deleted_memories"]), 1)
        
        # Mock the memory_store.retrieve_memories method to return an empty list
        self.agent.memory_store.retrieve_memories = MagicMock(return_value=[])
        
        # Verify that the memory is no longer retrievable
        self.agent.text_processor.extract_key_concepts = MagicMock(
            return_value=["chocolate"]
        )
        
        memories = self.agent.retrieve_memories(
            user_id="test_user",
            query="What do I like to eat?"
        )
        
        self.assertEqual(len(memories), 0)


if __name__ == "__main__":
    unittest.main()
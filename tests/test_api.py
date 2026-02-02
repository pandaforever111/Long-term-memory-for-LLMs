#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for the API module.
"""

import unittest
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Add the parent directory to the path so we can import the src modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api import app, get_memory_agent
from src.memory_agent import MemoryAgent


class TestAPI(unittest.TestCase):
    """Test cases for the API module."""

    def setUp(self):
        """Set up a test client and mock MemoryAgent for testing."""
        # Create a temporary database file
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp()
        
        # Create a mock MemoryAgent
        self.mock_memory_agent = MagicMock(spec=MemoryAgent)
        
        # Set up the mock responses
        self.mock_memory_agent.process_message.return_value = {
            "stored_memories": [
                {"content": "Test memory 1", "importance": 0.8},
                {"content": "Test memory 2", "importance": 0.6}
            ],
            "deleted_memories": []
        }
        
        self.mock_memory_agent.retrieve_memories.return_value = [
            {"content": "Test memory 1", "importance": 0.8, "created_at": "2023-01-01", "access_count": 1},
            {"content": "Test memory 2", "importance": 0.6, "created_at": "2023-01-02", "access_count": 2}
        ]
        
        self.mock_memory_agent.generate_response.return_value = {
            "response": "This is a test response.",
            "memories_used": [
                {"content": "Test memory 1", "importance": 0.8},
                {"content": "Test memory 2", "importance": 0.6}
            ]
        }
        
        self.mock_memory_agent.delete_memories.return_value = 2
        
        self.mock_memory_agent.get_memory_stats.return_value = {
            "total_memories": 2,
            "average_importance": 0.7,
            "most_accessed_memory": {
                "content": "Test memory 2",
                "importance": 0.6,
                "access_count": 2
            }
        }
        
        # Patch the get_memory_agent function to return our mock
        patcher = patch('src.api.get_memory_agent', return_value=self.mock_memory_agent)
        self.addCleanup(patcher.stop)
        patcher.start()
        
        # Create a test client
        self.client = TestClient(app)

    def tearDown(self):
        """Clean up the temporary database after testing."""
        os.close(self.temp_db_fd)
        os.unlink(self.temp_db_path)

    def test_health_check(self):
        """Test the health check endpoint."""
        response = self.client.get("/health")
        
        # Check the response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_generate_response(self):
        """Test the generate response endpoint."""
        # Test data
        request_data = {
            "user_id": "test_user",
            "message": "Hello, how are you?",
            "conversation_id": "test_conversation"
        }
        
        # Make the request
        response = self.client.post("/generate", json=request_data)
        
        # Check the response
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertEqual(response_data["response"], "This is a test response.")
        self.assertEqual(len(response_data["memories_used"]), 2)
        
        # Check that the MemoryAgent was called with the correct parameters
        self.mock_memory_agent.generate_response.assert_called_once_with(
            user_id="test_user",
            message="Hello, how are you?",
            conversation_id="test_conversation"
        )

    def test_process_message(self):
        """Test the process message endpoint."""
        # Test data
        request_data = {
            "user_id": "test_user",
            "message": "My name is John and I like pizza.",
            "conversation_id": "test_conversation"
        }
        
        # Make the request
        response = self.client.post("/process", json=request_data)
        
        # Check the response
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertEqual(len(response_data["stored_memories"]), 2)
        self.assertEqual(len(response_data["deleted_memories"]), 0)
        
        # Check that the MemoryAgent was called with the correct parameters
        self.mock_memory_agent.process_message.assert_called_once_with(
            user_id="test_user",
            message="My name is John and I like pizza.",
            conversation_id="test_conversation"
        )

    def test_retrieve_memories(self):
        """Test the retrieve memories endpoint."""
        # Test data
        request_data = {
            "user_id": "test_user",
            "query": "What do I like?",
            "limit": 5
        }
        
        # Make the request
        response = self.client.post("/memories", json=request_data)
        
        # Check the response
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertEqual(len(response_data["memories"]), 2)
        
        # Check that the MemoryAgent was called with the correct parameters
        self.mock_memory_agent.retrieve_memories.assert_called_once_with(
            user_id="test_user",
            query="What do I like?",
            limit=5
        )

    def test_delete_memories(self):
        """Test the delete memories endpoint."""
        # Test data
        request_data = {
            "user_id": "test_user",
            "content_pattern": "pizza"
        }
        
        # Make the request
        response = self.client.post("/memories/delete", json=request_data)
        
        # Check the response
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertEqual(response_data["deleted_count"], 2)
        
        # Check that the MemoryAgent was called with the correct parameters
        self.mock_memory_agent.delete_memories.assert_called_once_with(
            user_id="test_user",
            content_pattern="pizza"
        )

    def test_get_stats(self):
        """Test the get stats endpoint."""
        # Test data
        request_data = {
            "user_id": "test_user"
        }
        
        # Make the request
        response = self.client.post("/stats", json=request_data)
        
        # Check the response
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        self.assertEqual(response_data["total_memories"], 2)
        self.assertEqual(response_data["average_importance"], 0.7)
        self.assertEqual(response_data["most_accessed_memory"]["content"], "Test memory 2")
        
        # Check that the MemoryAgent was called with the correct parameters
        self.mock_memory_agent.get_memory_stats.assert_called_once_with(
            user_id="test_user"
        )

    def test_missing_user_id(self):
        """Test endpoints with missing user_id."""
        # Test data without user_id
        request_data = {
            "message": "Hello, how are you?"
        }
        
        # Test the generate endpoint
        response = self.client.post("/generate", json=request_data)
        self.assertEqual(response.status_code, 422)  # Unprocessable Entity
        
        # Test the process endpoint
        response = self.client.post("/process", json=request_data)
        self.assertEqual(response.status_code, 422)  # Unprocessable Entity
        
        # Test the memories endpoint
        request_data = {"query": "What do I like?"}
        response = self.client.post("/memories", json=request_data)
        self.assertEqual(response.status_code, 422)  # Unprocessable Entity
        
        # Test the delete endpoint
        request_data = {"content_pattern": "pizza"}
        response = self.client.post("/memories/delete", json=request_data)
        self.assertEqual(response.status_code, 422)  # Unprocessable Entity
        
        # Test the stats endpoint
        response = self.client.post("/stats", json={})
        self.assertEqual(response.status_code, 422)  # Unprocessable Entity


if __name__ == "__main__":
    unittest.main()
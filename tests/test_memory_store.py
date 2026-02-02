#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for the MemoryStore class.
"""

import unittest
import os
import tempfile
import sqlite3
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import the src modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory_store import MemoryStore


class TestMemoryStore(unittest.TestCase):
    """Test cases for the MemoryStore class."""

    def setUp(self):
        """Set up a temporary database for testing."""
        # Create a temporary file for the database
        self.temp_db_fd, self.temp_db_path = tempfile.mkstemp()
        self.memory_store = MemoryStore(self.temp_db_path)
        
        # Initialize the database
        self.memory_store.initialize_db()

    def tearDown(self):
        """Clean up the temporary database after testing."""
        # Close the database connection
        self.memory_store.close()
        
        # Close and remove the temporary file
        os.close(self.temp_db_fd)
        os.unlink(self.temp_db_path)

    def test_store_memory(self):
        """Test storing a memory in the database."""
        # Store a test memory
        user_id = "test_user"
        content = "This is a test memory"
        importance = 0.75
        concepts = ["test", "memory"]
        
        memory_id = self.memory_store.store_memory(
            user_id=user_id,
            content=content,
            importance=importance,
            concepts=concepts
        )
        
        # Verify the memory was stored
        self.assertIsNotNone(memory_id)
        
        # Retrieve the memory directly from the database
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT user_id, content, importance FROM memories WHERE id = ?", (memory_id,))
        result = cursor.fetchone()
        conn.close()
        
        # Check the retrieved memory matches what we stored
        self.assertEqual(result[0], user_id)
        self.assertEqual(result[1], content)
        self.assertEqual(result[2], importance)
        
        # Check that concepts were stored
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        cursor.execute(
            """SELECT c.name FROM concepts c 
            JOIN memory_concepts mc ON c.id = mc.concept_id 
            WHERE mc.memory_id = ?""", 
            (memory_id,)
        )
        stored_concepts = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        # Check that all concepts were stored
        for concept in concepts:
            self.assertIn(concept, stored_concepts)

    def test_retrieve_memories_by_query(self):
        """Test retrieving memories based on a query."""
        user_id = "test_user"
        
        # Store some test memories
        self.memory_store.store_memory(
            user_id=user_id,
            content="I like to play tennis on weekends",
            importance=0.8,
            concepts=["tennis", "sports", "weekends"]
        )
        
        self.memory_store.store_memory(
            user_id=user_id,
            content="My favorite color is blue",
            importance=0.6,
            concepts=["color", "preference"]
        )
        
        self.memory_store.store_memory(
            user_id=user_id,
            content="I enjoy playing basketball with friends",
            importance=0.7,
            concepts=["basketball", "sports", "friends"]
        )
        
        # Retrieve memories related to sports
        memories = self.memory_store.retrieve_memories_by_query(
            user_id=user_id,
            query="What sports do I like?",
            limit=5
        )
        
        # Check that we got the expected memories
        self.assertEqual(len(memories), 2)  # Should find the tennis and basketball memories
        
        # Check that the contents are as expected
        contents = [memory["content"] for memory in memories]
        self.assertIn("I like to play tennis on weekends", contents)
        self.assertIn("I enjoy playing basketball with friends", contents)

    def test_retrieve_memories_by_concepts(self):
        """Test retrieving memories based on concepts."""
        user_id = "test_user"
        
        # Store some test memories
        self.memory_store.store_memory(
            user_id=user_id,
            content="I like to play tennis on weekends",
            importance=0.8,
            concepts=["tennis", "sports", "weekends"]
        )
        
        self.memory_store.store_memory(
            user_id=user_id,
            content="My favorite color is blue",
            importance=0.6,
            concepts=["color", "preference"]
        )
        
        # Retrieve memories related to the "sports" concept
        memories = self.memory_store.retrieve_memories_by_concepts(
            user_id=user_id,
            concepts=["sports"],
            limit=5
        )
        
        # Check that we got the expected memory
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["content"], "I like to play tennis on weekends")

    def test_update_memory_access(self):
        """Test updating memory access statistics."""
        user_id = "test_user"
        
        # Store a test memory
        memory_id = self.memory_store.store_memory(
            user_id=user_id,
            content="This is a test memory",
            importance=0.5,
            concepts=["test"]
        )
        
        # Update access statistics
        self.memory_store.update_memory_access(memory_id)
        
        # Retrieve the memory directly from the database
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT access_count FROM memories WHERE id = ?", (memory_id,))
        access_count = cursor.fetchone()[0]
        conn.close()
        
        # Check that the access count was incremented
        self.assertEqual(access_count, 1)

    def test_delete_memories(self):
        """Test deleting memories based on content pattern."""
        user_id = "test_user"
        
        # Store some test memories
        self.memory_store.store_memory(
            user_id=user_id,
            content="I like apples",
            importance=0.5,
            concepts=["food", "preference"]
        )
        
        self.memory_store.store_memory(
            user_id=user_id,
            content="I like oranges",
            importance=0.5,
            concepts=["food", "preference"]
        )
        
        self.memory_store.store_memory(
            user_id=user_id,
            content="I dislike bananas",
            importance=0.5,
            concepts=["food", "preference"]
        )
        
        # Delete memories containing "like"
        deleted_count = self.memory_store.delete_memories(
            user_id=user_id,
            content_pattern="like"
        )
        
        # Check that two memories were deleted
        self.assertEqual(deleted_count, 2)
        
        # Retrieve all memories for the user
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT content FROM memories WHERE user_id = ?", (user_id,))
        remaining_memories = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        # Check that only the "dislike bananas" memory remains
        self.assertEqual(len(remaining_memories), 1)
        self.assertEqual(remaining_memories[0], "I dislike bananas")

    def test_prune_old_memories(self):
        """Test pruning old memories."""
        user_id = "test_user"
        
        # Store some test memories
        memory_id1 = self.memory_store.store_memory(
            user_id=user_id,
            content="This is an old memory",
            importance=0.3,  # Low importance
            concepts=["test"]
        )
        
        memory_id2 = self.memory_store.store_memory(
            user_id=user_id,
            content="This is an important memory",
            importance=0.9,  # High importance
            concepts=["test"]
        )
        
        # Manually update the creation date of the first memory to be old
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        old_date = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute(
            "UPDATE memories SET created_at = ? WHERE id = ?", 
            (old_date, memory_id1)
        )
        conn.commit()
        conn.close()
        
        # Prune memories older than 30 days with importance < 0.5
        pruned_count = self.memory_store.prune_old_memories(
            retention_days=30,
            importance_threshold=0.5
        )
        
        # Check that one memory was pruned
        self.assertEqual(pruned_count, 1)
        
        # Retrieve all memories for the user
        conn = sqlite3.connect(self.temp_db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT content FROM memories WHERE user_id = ?", (user_id,))
        remaining_memories = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        # Check that only the important memory remains
        self.assertEqual(len(remaining_memories), 1)
        self.assertEqual(remaining_memories[0], "This is an important memory")

    def test_get_memory_stats(self):
        """Test getting memory statistics for a user."""
        user_id = "test_user"
        
        # Store some test memories with different importance values
        memory_id1 = self.memory_store.store_memory(
            user_id=user_id,
            content="Memory 1",
            importance=0.3,
            concepts=["test"]
        )
        
        memory_id2 = self.memory_store.store_memory(
            user_id=user_id,
            content="Memory 2",
            importance=0.7,
            concepts=["test"]
        )
        
        # Update access count for the second memory
        self.memory_store.update_memory_access(memory_id2)
        self.memory_store.update_memory_access(memory_id2)  # Access twice
        
        # Get memory statistics
        stats = self.memory_store.get_memory_stats(user_id)
        
        # Check the statistics
        self.assertEqual(stats["total_memories"], 2)
        self.assertAlmostEqual(stats["average_importance"], 0.5, places=1)  # (0.3 + 0.7) / 2 = 0.5
        self.assertEqual(stats["most_accessed_memory"]["content"], "Memory 2")
        self.assertEqual(stats["most_accessed_memory"]["access_count"], 2)


if __name__ == "__main__":
    unittest.main()
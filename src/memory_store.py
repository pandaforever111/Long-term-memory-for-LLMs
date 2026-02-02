#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Memory Store Module

This module handles the storage, retrieval, and management of memories for the GPT Memory Agent.
"""

import os
import json
import sqlite3
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime
import uuid

from src.config import Config


class MemoryStore:
    """Memory storage and retrieval system for the GPT Memory Agent."""

    def __init__(self, config: Config):
        """Initialize the memory store with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger("memory_agent.memory_store")
        self.db_path = self.config.db_path
        self._initialize_database()

    def _initialize_database(self) -> None:
        """Initialize the SQLite database and create necessary tables if they don't exist."""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        self.logger.info(f"Initializing database at {self.db_path}")
        
        # Connect to the database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create memories table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            content TEXT NOT NULL,
            source_message TEXT,
            conversation_id TEXT,
            embedding TEXT,
            importance REAL DEFAULT 0.5,
            created_at TIMESTAMP NOT NULL,
            last_accessed TIMESTAMP,
            access_count INTEGER DEFAULT 0,
            metadata TEXT
        )
        """)
        
        # Create concepts table for semantic indexing
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS concepts (
            id TEXT PRIMARY KEY,
            concept TEXT NOT NULL,
            UNIQUE(concept)
        )
        """)
        
        # Create memory_concepts table for many-to-many relationship
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS memory_concepts (
            memory_id TEXT,
            concept_id TEXT,
            weight REAL DEFAULT 1.0,
            PRIMARY KEY (memory_id, concept_id),
            FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE,
            FOREIGN KEY (concept_id) REFERENCES concepts(id) ON DELETE CASCADE
        )
        """)
        
        # Create index on user_id for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories(user_id)")
        
        # Create index on conversation_id for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_memories_conversation_id ON memories(conversation_id)")
        
        # Commit changes and close connection
        conn.commit()
        conn.close()
        
        self.logger.debug("Database initialized successfully")

    def store_memory(
        self,
        user_id: str,
        content: str,
        source_message: Optional[str] = None,
        conversation_id: Optional[str] = None,
        timestamp: Optional[datetime] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5
    ) -> str:
        """Store a new memory in the database.
        
        Args:
            user_id: Unique identifier for the user
            content: The memory content to store
            source_message: Original message that generated this memory
            conversation_id: Identifier for the conversation
            timestamp: When the memory was created (defaults to now)
            metadata: Additional metadata for the memory
            importance: Initial importance score (0.0 to 1.0)
            
        Returns:
            The ID of the stored memory
        """
        # Generate a unique ID for the memory
        memory_id = str(uuid.uuid4())
        
        # Set timestamp to now if not provided
        if timestamp is None:
            timestamp = datetime.now()
        
        # Convert metadata to JSON string if provided
        metadata_json = json.dumps(metadata) if metadata else None
        
        # Connect to the database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Insert the memory
            cursor.execute("""
            INSERT INTO memories (
                id, user_id, content, source_message, conversation_id,
                importance, created_at, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                memory_id, user_id, content, source_message, conversation_id,
                importance, timestamp.isoformat(), metadata_json
            ))
            
            # Extract concepts from the memory content and store them
            # This would typically use the TextProcessor, but for simplicity,
            # we'll just use a placeholder here
            # In a real implementation, you would import and use the TextProcessor
            concepts = self._extract_concepts(content)
            
            for concept, weight in concepts:
                # Get or create concept ID
                concept_id = self._get_or_create_concept(cursor, concept)
                
                # Link memory to concept with weight
                cursor.execute("""
                INSERT INTO memory_concepts (memory_id, concept_id, weight)
                VALUES (?, ?, ?)
                """, (memory_id, concept_id, weight))
            
            # Commit the transaction
            conn.commit()
            self.logger.debug(f"Stored memory {memory_id} for user {user_id}")
            
            return memory_id
            
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error storing memory: {e}")
            raise
            
        finally:
            conn.close()

    def _extract_concepts(self, content: str) -> List[Tuple[str, float]]:
        """Extract key concepts from memory content with weights.
        
        This is a placeholder implementation. In a real system, this would use
        the TextProcessor to extract meaningful concepts.
        
        Args:
            content: The memory content
            
        Returns:
            List of (concept, weight) tuples
        """
        # Simple placeholder implementation - split by spaces and assign equal weights
        # In a real implementation, this would use NLP techniques
        words = [word.lower() for word in content.split() if len(word) > 3]
        return [(word, 1.0) for word in set(words)]

    def _get_or_create_concept(self, cursor: sqlite3.Cursor, concept: str) -> str:
        """Get an existing concept ID or create a new one.
        
        Args:
            cursor: Database cursor
            concept: The concept text
            
        Returns:
            Concept ID
        """
        # Check if concept exists
        cursor.execute("SELECT id FROM concepts WHERE concept = ?", (concept,))
        result = cursor.fetchone()
        
        if result:
            return result[0]
        
        # Create new concept
        concept_id = str(uuid.uuid4())
        cursor.execute("INSERT INTO concepts (id, concept) VALUES (?, ?)", (concept_id, concept))
        return concept_id

    def retrieve_memories(
        self,
        user_id: str,
        query: str,
        concepts: Optional[List[str]] = None,
        limit: int = 5,
        min_importance: float = 0.0,
        conversation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve relevant memories based on query and concepts.
        
        Args:
            user_id: Unique identifier for the user
            query: The query text to match against memories
            concepts: List of key concepts to match
            limit: Maximum number of memories to retrieve
            min_importance: Minimum importance score for memories
            conversation_id: Optional filter by conversation
            
        Returns:
            List of memory objects
        """
        self.logger.info(f"Retrieving memories for user {user_id} with query: {query}")
        
        # Connect to the database
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        cursor = conn.cursor()
        
        try:
            # Base query parts
            select_clause = """
            SELECT m.id, m.content, m.source_message, m.conversation_id, 
                   m.importance, m.created_at, m.metadata, 
                   COUNT(mc.concept_id) as concept_matches,
                   SUM(mc.weight) as relevance_score
            FROM memories m
            """
            
            where_clause = "WHERE m.user_id = ?"
            params = [user_id]
            
            # Add conversation filter if provided
            if conversation_id:
                where_clause += " AND m.conversation_id = ?"
                params.append(conversation_id)
            
            # Add importance filter
            where_clause += " AND m.importance >= ?"
            params.append(min_importance)
            
            # If we have concepts, join with memory_concepts and filter
            join_clause = ""
            if concepts and len(concepts) > 0:
                concept_placeholders = ", ".join(["?" for _ in concepts])
                join_clause = f"""
                LEFT JOIN memory_concepts mc ON m.id = mc.memory_id
                LEFT JOIN concepts c ON mc.concept_id = c.id AND c.concept IN ({concept_placeholders})
                """
                params.extend(concepts)
                
                # Group by memory ID to aggregate concept matches
                group_by = "GROUP BY m.id"
                
                # Order by concept matches and relevance score
                order_by = "ORDER BY concept_matches DESC, relevance_score DESC, m.importance DESC"
            else:
                # If no concepts, just use full-text search on content
                where_clause += " AND m.content LIKE ?"
                params.append(f"%{query}%")
                
                # No need for grouping
                group_by = ""
                
                # Order by importance and recency
                order_by = "ORDER BY m.importance DESC, m.created_at DESC"
            
            # Combine all parts of the query
            full_query = f"{select_clause} {join_clause} {where_clause} {group_by} {order_by} LIMIT ?"
            params.append(limit)
            
            # Execute the query
            cursor.execute(full_query, params)
            rows = cursor.fetchall()
            
            # Convert rows to dictionaries
            memories = []
            for row in rows:
                memory = dict(row)
                
                # Parse metadata JSON if present
                if memory['metadata']:
                    memory['metadata'] = json.loads(memory['metadata'])
                
                # Update access statistics
                self._update_memory_access(cursor, memory['id'])
                
                memories.append(memory)
            
            # Commit the access updates
            conn.commit()
            
            self.logger.debug(f"Retrieved {len(memories)} memories")
            return memories
            
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error retrieving memories: {e}")
            raise
            
        finally:
            conn.close()

    def _update_memory_access(self, cursor: sqlite3.Cursor, memory_id: str) -> None:
        """Update the access statistics for a memory.
        
        Args:
            cursor: Database cursor
            memory_id: The memory ID to update
        """
        cursor.execute("""
        UPDATE memories 
        SET last_accessed = ?, access_count = access_count + 1
        WHERE id = ?
        """, (datetime.now().isoformat(), memory_id))

    def delete_memories_by_content(self, user_id: str, content_pattern: str) -> List[str]:
        """Delete memories matching a content pattern for a user.
        
        Args:
            user_id: Unique identifier for the user
            content_pattern: Pattern to match against memory content
            
        Returns:
            List of deleted memory IDs
        """
        self.logger.info(f"Deleting memories for user {user_id} matching: {content_pattern}")
        
        # Connect to the database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # First, get the IDs of memories to delete
            cursor.execute("""
            SELECT id FROM memories 
            WHERE user_id = ? AND content LIKE ?
            """, (user_id, f"%{content_pattern}%"))
            
            memory_ids = [row[0] for row in cursor.fetchall()]
            
            if memory_ids:
                # Delete the memories
                placeholders = ", ".join(["?" for _ in memory_ids])
                cursor.execute(f"""
                DELETE FROM memories 
                WHERE id IN ({placeholders})
                """, memory_ids)
                
                # Commit the transaction
                conn.commit()
                
                self.logger.debug(f"Deleted {len(memory_ids)} memories")
                return memory_ids
            else:
                self.logger.debug("No memories found matching the pattern")
                return []
            
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error deleting memories: {e}")
            raise
            
        finally:
            conn.close()

    def get_memory_by_id(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a specific memory by its ID.
        
        Args:
            memory_id: The ID of the memory to retrieve
            
        Returns:
            Memory object or None if not found
        """
        # Connect to the database
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
            SELECT id, user_id, content, source_message, conversation_id,
                   importance, created_at, last_accessed, access_count, metadata
            FROM memories 
            WHERE id = ?
            """, (memory_id,))
            
            row = cursor.fetchone()
            
            if row:
                memory = dict(row)
                
                # Parse metadata JSON if present
                if memory['metadata']:
                    memory['metadata'] = json.loads(memory['metadata'])
                
                # Update access statistics
                self._update_memory_access(cursor, memory_id)
                conn.commit()
                
                return memory
            else:
                return None
            
        except Exception as e:
            self.logger.error(f"Error retrieving memory {memory_id}: {e}")
            return None
            
        finally:
            conn.close()

    def update_memory_importance(self, memory_id: str, importance: float) -> bool:
        """Update the importance score of a memory.
        
        Args:
            memory_id: The ID of the memory to update
            importance: New importance score (0.0 to 1.0)
            
        Returns:
            True if successful, False otherwise
        """
        # Validate importance score
        importance = max(0.0, min(1.0, importance))
        
        # Connect to the database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
            UPDATE memories 
            SET importance = ?
            WHERE id = ?
            """, (importance, memory_id))
            
            conn.commit()
            success = cursor.rowcount > 0
            
            if success:
                self.logger.debug(f"Updated importance of memory {memory_id} to {importance}")
            else:
                self.logger.warning(f"Memory {memory_id} not found for importance update")
                
            return success
            
        except Exception as e:
            conn.rollback()
            self.logger.error(f"Error updating memory importance: {e}")
            return False
            
        finally:
            conn.close()

    def get_user_memory_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics about a user's memories.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            Dictionary of memory statistics
        """
        # Connect to the database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get total count
            cursor.execute("SELECT COUNT(*) FROM memories WHERE user_id = ?", (user_id,))
            total_count = cursor.fetchone()[0]
            
            # Get average importance
            cursor.execute("SELECT AVG(importance) FROM memories WHERE user_id = ?", (user_id,))
            avg_importance = cursor.fetchone()[0] or 0.0
            
            # Get oldest memory date
            cursor.execute("SELECT MIN(created_at) FROM memories WHERE user_id = ?", (user_id,))
            oldest_date = cursor.fetchone()[0]
            
            # Get newest memory date
            cursor.execute("SELECT MAX(created_at) FROM memories WHERE user_id = ?", (user_id,))
            newest_date = cursor.fetchone()[0]
            
            # Get most accessed memory
            cursor.execute("""
            SELECT id, content, access_count 
            FROM memories 
            WHERE user_id = ? 
            ORDER BY access_count DESC 
            LIMIT 1
            """, (user_id,))
            most_accessed = cursor.fetchone()
            
            return {
                "user_id": user_id,
                "total_memories": total_count,
                "average_importance": avg_importance,
                "oldest_memory_date": oldest_date,
                "newest_memory_date": newest_date,
                "most_accessed_memory": {
                    "id": most_accessed[0],
                    "content": most_accessed[1],
                    "access_count": most_accessed[2]
                } if most_accessed else None
            }
            
        except Exception as e:
            self.logger.error(f"Error getting user memory stats: {e}")
            return {
                "user_id": user_id,
                "error": str(e)
            }
            
        finally:
            conn.close()
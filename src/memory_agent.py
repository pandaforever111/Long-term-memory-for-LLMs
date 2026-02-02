#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Memory Agent for GPT

This module implements the main memory agent that integrates with OpenAI's GPT models
to provide long-term memory capabilities.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory_store import MemoryStore
from src.ai_client_factory import AIClientFactory
from src.text_processor import TextProcessor
from src.config import Config


class MemoryAgent:
    """Main memory agent class that orchestrates memory operations and OpenAI API interactions."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the memory agent with configuration.
        
        Args:
            config_path: Path to the configuration file. If None, default config is used.
        """
        self.config = Config(config_path)
        self.logger = self._setup_logger()
        self.memory_store = MemoryStore(self.config)
        self.ai_client = AIClientFactory.create_client(self.config)
        self.text_processor = TextProcessor(self.config)
        self.logger.info(f"Memory agent initialized with {self.config.ai_provider} AI provider")

    def _setup_logger(self) -> logging.Logger:
        """Set up and configure the logger."""
        logger = logging.getLogger("memory_agent")
        logger.setLevel(self.config.log_level)
        
        # Create console handler
        handler = logging.StreamHandler()
        handler.setLevel(self.config.log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
        
        return logger

    def process_message(self, user_id: str, message: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a user message, extract memories, and store them if applicable.
        
        Args:
            user_id: Unique identifier for the user
            message: The user's message text
            conversation_id: Optional conversation identifier
            
        Returns:
            Dict containing processing results and any extracted memories
        """
        self.logger.info(f"Processing message for user {user_id}")
        
        # Generate a conversation ID if not provided
        if not conversation_id:
            conversation_id = f"conv_{datetime.now().strftime('%Y%m%d%H%M%S')}_{user_id}"
        
        # Extract potential memories from the message
        memory_candidates = self.text_processor.extract_memory_candidates(message)
        
        # Filter and store valid memories
        stored_memories = []
        for candidate in memory_candidates:
            # Determine if this is actually a memory worth storing
            if self.text_processor.is_valid_memory(candidate):
                memory_id = self.memory_store.store_memory(
                    user_id=user_id,
                    content=candidate,
                    source_message=message,
                    conversation_id=conversation_id,
                    timestamp=datetime.now()
                )
                stored_memories.append({
                    "memory_id": memory_id,
                    "content": candidate
                })
                self.logger.debug(f"Stored memory: {candidate}")
        
        # Check if the message is requesting memory deletion
        deletion_requests = self.text_processor.extract_deletion_requests(message)
        deleted_memories = []
        
        for deletion_request in deletion_requests:
            deleted_memory_ids = self.memory_store.delete_memories_by_content(
                user_id=user_id,
                content_pattern=deletion_request
            )
            for memory_id in deleted_memory_ids:
                deleted_memories.append({
                    "memory_id": memory_id,
                    "deletion_pattern": deletion_request
                })
                self.logger.debug(f"Deleted memory with ID {memory_id}")
        
        return {
            "user_id": user_id,
            "conversation_id": conversation_id,
            "stored_memories": stored_memories,
            "deleted_memories": deleted_memories,
            "timestamp": datetime.now().isoformat()
        }

    def retrieve_memories(self, user_id: str, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant memories for a user based on a query.
        
        Args:
            user_id: Unique identifier for the user
            query: The query to search for relevant memories
            limit: Maximum number of memories to retrieve
            
        Returns:
            List of relevant memories
        """
        self.logger.info(f"Retrieving memories for user {user_id} with query: {query}")
        
        # Process the query to extract key concepts
        query_concepts = self.text_processor.extract_key_concepts(query)
        
        # Retrieve memories from the store
        memories = self.memory_store.retrieve_memories(
            user_id=user_id,
            query=query,
            concepts=query_concepts,
            limit=limit
        )
        
        self.logger.debug(f"Retrieved {len(memories)} memories")
        return memories

    def generate_response(self, user_id: str, message: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate a response using GPT with relevant memories included.
        
        Args:
            user_id: Unique identifier for the user
            message: The user's message text
            conversation_id: Optional conversation identifier
            
        Returns:
            Dict containing the generated response and processing information
        """
        self.logger.info(f"Generating response for user {user_id}")
        
        # Process the message to store any new memories
        process_result = self.process_message(user_id, message, conversation_id)
        
        # Retrieve relevant memories for this message
        memories = self.retrieve_memories(user_id, message)
        
        # Format memories for inclusion in the prompt
        memory_context = self._format_memories_for_prompt(memories)
        
        # Generate response using AI client with memory context
        response = self.ai_client.generate_response(
            message=message,
            memory_context=memory_context,
            user_id=user_id
        )
        
        return {
            "user_id": user_id,
            "conversation_id": process_result["conversation_id"],
            "message": message,
            "response": response,
            "memories_used": memories,
            "memories_stored": process_result["stored_memories"],
            "memories_deleted": process_result["deleted_memories"],
            "timestamp": datetime.now().isoformat()
        }

    def _format_memories_for_prompt(self, memories: List[Dict[str, Any]]) -> str:
        """Format retrieved memories for inclusion in the prompt.
        
        Args:
            memories: List of memory objects
            
        Returns:
            Formatted memory context string
        """
        if not memories:
            return ""
        
        memory_strings = [f"- {memory['content']}" for memory in memories]
        formatted_context = "Based on our previous conversations, I recall the following information about you:\n"
        formatted_context += "\n".join(memory_strings)
        
        return formatted_context


def main():
    """Main function to run the memory agent as a standalone application."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GPT Memory Agent")
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--user_id", 
        type=str, 
        default="test_user",
        help="User ID for testing"
    )
    parser.add_argument(
        "--interactive", 
        action="store_true",
        help="Run in interactive mode"
    )
    args = parser.parse_args()
    
    # Initialize the memory agent
    agent = MemoryAgent(args.config)
    
    if args.interactive:
        print("=== GPT Memory Agent Interactive Mode ===")
        print("Type 'exit' to quit")
        
        conversation_id = f"conv_{datetime.now().strftime('%Y%m%d%H%M%S')}_{args.user_id}"
        
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == "exit":
                break
            
            result = agent.generate_response(
                user_id=args.user_id,
                message=user_input,
                conversation_id=conversation_id
            )
            
            print(f"\nAI: {result['response']}")
            
            # Print debug info about memories if any were used
            if result['memories_used']:
                print("\n[Debug] Memories used:")
                for memory in result['memories_used']:
                    print(f"- {memory['content']}")
            
            # Print debug info about new memories if any were stored
            if result['memories_stored']:
                print("\n[Debug] New memories stored:")
                for memory in result['memories_stored']:
                    print(f"- {memory['content']}")
            
            # Print debug info about deleted memories if any were deleted
            if result['memories_deleted']:
                print("\n[Debug] Memories deleted:")
                for memory in result['memories_deleted']:
                    print(f"- {memory['deletion_pattern']}")
    else:
        # Run a simple test
        test_message = "I use Shram and Magnet as productivity tools"
        result = agent.process_message(args.user_id, test_message)
        print(f"Processed message: {test_message}")
        print(f"Stored memories: {result['stored_memories']}")
        
        # Test memory retrieval
        query = "What productivity tools do I use?"
        memories = agent.retrieve_memories(args.user_id, query)
        print(f"\nQuery: {query}")
        print(f"Retrieved memories: {memories}")


if __name__ == "__main__":
    main()
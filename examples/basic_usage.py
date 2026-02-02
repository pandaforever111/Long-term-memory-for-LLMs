#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Basic Usage Example for GPT Memory Agent

This script demonstrates basic usage of the GPT Memory Agent.
"""

import os
import sys
import logging
from datetime import datetime

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.memory_agent import MemoryAgent
from src.config import Config


def setup_logging():
    """Set up basic logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def run_example():
    """Run a basic example of the memory agent."""
    print("=== GPT Memory Agent Basic Example ===")
    
    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("\nError: OPENAI_API_KEY environment variable not set.")
        print("Please set your OpenAI API key as an environment variable:")
        print("  export OPENAI_API_KEY='your-api-key'  # Linux/macOS")
        print("  set OPENAI_API_KEY=your-api-key  # Windows")
        return
    
    # Initialize the memory agent
    print("\nInitializing memory agent...")
    agent = MemoryAgent()
    
    # Set up a test user and conversation
    user_id = "example_user"
    conversation_id = f"example_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    # Example 1: Store a memory
    print("\nExample 1: Storing a memory")
    message1 = "My name is Alice and I work as a software engineer."
    print(f"User: {message1}")
    
    result1 = agent.process_message(
        user_id=user_id,
        message=message1,
        conversation_id=conversation_id
    )
    
    print(f"Stored {len(result1['stored_memories'])} memories:")
    for memory in result1['stored_memories']:
        print(f"  - {memory['content']}")
    
    # Example 2: Store another memory
    print("\nExample 2: Storing another memory")
    message2 = "I really enjoy hiking in the mountains on weekends."
    print(f"User: {message2}")
    
    result2 = agent.process_message(
        user_id=user_id,
        message=message2,
        conversation_id=conversation_id
    )
    
    print(f"Stored {len(result2['stored_memories'])} memories:")
    for memory in result2['stored_memories']:
        print(f"  - {memory['content']}")
    
    # Example 3: Retrieve memories
    print("\nExample 3: Retrieving memories")
    query = "What do I like to do in my free time?"
    print(f"Query: {query}")
    
    memories = agent.retrieve_memories(
        user_id=user_id,
        query=query
    )
    
    print(f"Retrieved {len(memories)} memories:")
    for memory in memories:
        print(f"  - {memory['content']}")
    
    # Example 4: Generate a response with memory context
    print("\nExample 4: Generating a response with memory context")
    message3 = "What do you remember about me and what I like to do?"
    print(f"User: {message3}")
    
    result3 = agent.generate_response(
        user_id=user_id,
        message=message3,
        conversation_id=conversation_id
    )
    
    print(f"AI: {result3['response']}")
    print("\nMemories used:")
    for memory in result3['memories_used']:
        print(f"  - {memory['content']}")
    
    # Example 5: Delete a memory
    print("\nExample 5: Deleting a memory")
    message4 = "Please forget that I like hiking."
    print(f"User: {message4}")
    
    result4 = agent.process_message(
        user_id=user_id,
        message=message4,
        conversation_id=conversation_id
    )
    
    print(f"Deleted {len(result4['deleted_memories'])} memories")
    
    # Example 6: Verify memory was deleted
    print("\nExample 6: Verifying memory deletion")
    query = "What do I like to do?"
    print(f"Query: {query}")
    
    memories = agent.retrieve_memories(
        user_id=user_id,
        query=query
    )
    
    print(f"Retrieved {len(memories)} memories:")
    for memory in memories:
        print(f"  - {memory['content']}")
    
    # Example 7: Get memory statistics
    print("\nExample 7: Getting memory statistics")
    stats = agent.memory_store.get_user_memory_stats(user_id)
    
    print(f"Total memories: {stats['total_memories']}")
    print(f"Average importance: {stats['average_importance']:.2f}")
    if stats['most_accessed_memory']:
        print(f"Most accessed memory: {stats['most_accessed_memory']['content']}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    setup_logging()
    run_example()
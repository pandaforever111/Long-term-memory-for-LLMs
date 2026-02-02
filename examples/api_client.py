#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
API Client Example for GPT Memory Agent

This script demonstrates how to interact with the GPT Memory Agent API.
"""

import requests
import json
import sys
from typing import Dict, Any, Optional


class MemoryAgentClient:
    """Client for interacting with the GPT Memory Agent API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        """Initialize the API client.
        
        Args:
            base_url: Base URL of the Memory Agent API
        """
        self.base_url = base_url.rstrip('/')

    def generate_response(self, user_id: str, message: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate a response with memory context.
        
        Args:
            user_id: Unique identifier for the user
            message: The user's message
            conversation_id: Optional conversation identifier
            
        Returns:
            API response
        """
        url = f"{self.base_url}/generate"
        payload = {
            "user_id": user_id,
            "message": message
        }
        
        if conversation_id:
            payload["conversation_id"] = conversation_id
            
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def process_message(self, user_id: str, message: str, conversation_id: Optional[str] = None) -> Dict[str, Any]:
        """Process a message to extract and store memories.
        
        Args:
            user_id: Unique identifier for the user
            message: The user's message
            conversation_id: Optional conversation identifier
            
        Returns:
            API response
        """
        url = f"{self.base_url}/process"
        payload = {
            "user_id": user_id,
            "message": message
        }
        
        if conversation_id:
            payload["conversation_id"] = conversation_id
            
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def retrieve_memories(self, user_id: str, query: str, limit: int = 5) -> Dict[str, Any]:
        """Retrieve relevant memories for a user based on a query.
        
        Args:
            user_id: Unique identifier for the user
            query: Query to search for relevant memories
            limit: Maximum number of memories to retrieve
            
        Returns:
            API response
        """
        url = f"{self.base_url}/memories"
        payload = {
            "user_id": user_id,
            "query": query,
            "limit": limit
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def delete_memories(self, user_id: str, content_pattern: str) -> Dict[str, Any]:
        """Delete memories matching a content pattern.
        
        Args:
            user_id: Unique identifier for the user
            content_pattern: Pattern to match against memory content
            
        Returns:
            API response
        """
        url = f"{self.base_url}/memories/delete"
        payload = {
            "user_id": user_id,
            "content_pattern": content_pattern
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def get_stats(self, user_id: str) -> Dict[str, Any]:
        """Get memory statistics for a user.
        
        Args:
            user_id: Unique identifier for the user
            
        Returns:
            API response
        """
        url = f"{self.base_url}/stats"
        payload = {
            "user_id": user_id
        }
        
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()

    def health_check(self) -> Dict[str, Any]:
        """Check the health of the API.
        
        Returns:
            API response
        """
        url = f"{self.base_url}/health"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()


def run_example():
    """Run an example of using the API client."""
    print("=== GPT Memory Agent API Client Example ===")
    
    # Initialize the client
    client = MemoryAgentClient("http://localhost:8000")
    
    # Check if the API is running
    try:
        health = client.health_check()
        print(f"\nAPI Status: {health['status']}")
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to the API. Make sure it's running.")
        print("Start the API with: python -m src.api")
        return
    
    # Set up a test user
    user_id = "api_example_user"
    
    # Example 1: Store a memory
    print("\nExample 1: Storing a memory")
    message1 = "My name is Bob and I'm a graphic designer."
    print(f"User: {message1}")
    
    try:
        result1 = client.process_message(user_id, message1)
        print(f"Stored {len(result1['stored_memories'])} memories:")
        for memory in result1['stored_memories']:
            print(f"  - {memory['content']}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 2: Store another memory
    print("\nExample 2: Storing another memory")
    message2 = "I love playing tennis on the weekends."
    print(f"User: {message2}")
    
    try:
        result2 = client.process_message(user_id, message2)
        print(f"Stored {len(result2['stored_memories'])} memories:")
        for memory in result2['stored_memories']:
            print(f"  - {memory['content']}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 3: Retrieve memories
    print("\nExample 3: Retrieving memories")
    query = "What do I do for work?"
    print(f"Query: {query}")
    
    try:
        result3 = client.retrieve_memories(user_id, query)
        print(f"Retrieved {len(result3['memories'])} memories:")
        for memory in result3['memories']:
            print(f"  - {memory['content']}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 4: Generate a response
    print("\nExample 4: Generating a response")
    message3 = "Can you tell me what you know about me?"
    print(f"User: {message3}")
    
    try:
        result4 = client.generate_response(user_id, message3)
        print(f"AI: {result4['response']}")
        print("\nMemories used:")
        for memory in result4['memories_used']:
            print(f"  - {memory['content']}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Example 5: Get memory statistics
    print("\nExample 5: Getting memory statistics")
    
    try:
        stats = client.get_stats(user_id)
        print(f"Total memories: {stats['total_memories']}")
        print(f"Average importance: {stats['average_importance']:.2f}")
        if stats.get('most_accessed_memory'):
            print(f"Most accessed memory: {stats['most_accessed_memory']['content']}")
    except Exception as e:
        print(f"Error: {e}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    run_example()
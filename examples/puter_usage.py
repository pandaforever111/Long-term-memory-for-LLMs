#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Puter.js Client Usage Example

This example demonstrates how to use the PuterClient with the Memory Agent.
Note: This is a demonstration only. In a real implementation, you would need to
integrate with the actual Puter.js JavaScript library in a web application.
"""

import os
import sys
import logging
from pathlib import Path

# Add the parent directory to the path so we can import from src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config
from src.memory_agent import MemoryAgent

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Run the example."""
    # Create a custom configuration with Puter.js as the AI provider
    config = Config()
    
    # Set the AI provider to Puter
    config.ai_provider = "puter"
    
    # Initialize the memory agent with the custom configuration
    agent = MemoryAgent(config=config)
    
    # Define a user ID for this session
    user_id = "puter_demo_user"
    
    # Process a message to store memories
    logger.info("Processing message to store memories...")
    result = agent.process_message(user_id, "My name is Alex and I work as a software engineer.")
    logger.info(f"Stored {len(result['stored_memories'])} memories")
    
    # Generate a response with memory context
    logger.info("\nGenerating response with memory context...")
    response = agent.generate_response(user_id, "What do I do for a living?")
    logger.info(f"Response: {response['response']}")
    
    # Process another message
    logger.info("\nProcessing another message...")
    result = agent.process_message(user_id, "I have a dog named Max and I live in Seattle.")
    logger.info(f"Stored {len(result['stored_memories'])} memories")
    
    # Generate another response
    logger.info("\nGenerating another response...")
    response = agent.generate_response(user_id, "Tell me about myself.")
    logger.info(f"Response: {response['response']}")
    
    # Delete a memory
    logger.info("\nDeleting memory about living in Seattle...")
    result = agent.delete_memories(user_id, "Seattle")
    logger.info(f"Deleted {result['deleted_count']} memories")
    
    # Generate a final response
    logger.info("\nGenerating final response after deletion...")
    response = agent.generate_response(user_id, "Where do I live?")
    logger.info(f"Response: {response['response']}")
    
    logger.info("\nExample completed successfully!")
    logger.info("Note: This example uses a placeholder PuterClient implementation.")
    logger.info("In a real application, you would use the Puter.js web demo instead.")

if __name__ == "__main__":
    main()
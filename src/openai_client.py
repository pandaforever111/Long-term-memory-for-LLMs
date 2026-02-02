#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OpenAI Client Module

This module handles interactions with the OpenAI API for the GPT Memory Agent.
"""

import os
import logging
from typing import Dict, List, Optional, Union, Any
import json

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from src.config import Config


class OpenAIClient:
    """Client for interacting with OpenAI's API."""

    def __init__(self, config: Config):
        """Initialize the OpenAI client with configuration.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.logger = logging.getLogger("memory_agent.openai_client")
        
        # Check if OpenAI package is available
        if not OPENAI_AVAILABLE:
            self.logger.error("OpenAI package is not installed. Please install it with 'pip install openai'.")
            raise ImportError("OpenAI package is required but not installed")
        
        # Initialize the OpenAI client
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the OpenAI client with API key."""
        api_key = self.config.openai_api_key
        
        # If API key is not in config, try to get it from environment variable
        if not api_key:
            api_key = os.environ.get("OPENAI_API_KEY")
            
        if not api_key:
            self.logger.error("OpenAI API key not found in config or environment variables")
            raise ValueError("OpenAI API key is required but not provided")
        
        try:
            self.client = OpenAI(api_key=api_key)
            self.logger.info("OpenAI client initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize OpenAI client: {e}")
            raise

    def generate_response(
        self,
        message: str,
        memory_context: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_id: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate a response using the OpenAI API.
        
        Args:
            message: The user's message
            memory_context: Optional context from memory retrieval
            conversation_history: Optional conversation history
            user_id: Optional user identifier
            system_prompt: Optional system prompt override
            
        Returns:
            Generated response text
        """
        self.logger.info("Generating response with OpenAI API")
        
        # Prepare the messages for the API call
        messages = self._prepare_messages(
            message=message,
            memory_context=memory_context,
            conversation_history=conversation_history,
            system_prompt=system_prompt
        )
        
        # Log the messages being sent (excluding sensitive content in production)
        if self.config.log_level == logging.DEBUG:
            self.logger.debug(f"Sending messages to OpenAI API: {json.dumps(messages, indent=2)}")
        
        try:
            # Make the API call
            response = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p,
                frequency_penalty=self.config.frequency_penalty,
                presence_penalty=self.config.presence_penalty,
                user=user_id  # Pass user ID for OpenAI's monitoring
            )
            
            # Extract the response text
            response_text = response.choices[0].message.content
            
            self.logger.debug(f"Received response from OpenAI API: {response_text[:50]}...")
            return response_text
            
        except Exception as e:
            self.logger.error(f"Error generating response with OpenAI API: {e}")
            # Return a fallback response
            return "I'm sorry, I encountered an error while processing your request. Please try again later."

    def _prepare_messages(
        self,
        message: str,
        memory_context: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """Prepare messages for the OpenAI API call.
        
        Args:
            message: The user's message
            memory_context: Optional context from memory retrieval
            conversation_history: Optional conversation history
            system_prompt: Optional system prompt override
            
        Returns:
            List of message dictionaries for the API call
        """
        messages = []
        
        # Add system prompt
        if not system_prompt:
            system_prompt = self.config.system_prompt
            
        # If we have memory context, include it in the system prompt
        if memory_context:
            system_prompt = f"{system_prompt}\n\n{memory_context}"
            
        messages.append({"role": "system", "content": system_prompt})
        
        # Add conversation history if provided
        if conversation_history:
            messages.extend(conversation_history)
        
        # Add the current user message
        messages.append({"role": "user", "content": message})
        
        return messages

    def extract_embedding(
        self,
        text: str,
        model: Optional[str] = None
    ) -> List[float]:
        """Generate an embedding vector for the given text.
        
        Args:
            text: The text to generate an embedding for
            model: Optional embedding model name override
            
        Returns:
            List of embedding values
        """
        if not model:
            model = self.config.embedding_model
            
        try:
            response = self.client.embeddings.create(
                input=text,
                model=model
            )
            
            embedding = response.data[0].embedding
            self.logger.debug(f"Generated embedding with {len(embedding)} dimensions")
            return embedding
            
        except Exception as e:
            self.logger.error(f"Error generating embedding: {e}")
            # Return an empty embedding as fallback
            return []

    def moderate_content(
        self,
        text: str
    ) -> Dict[str, Any]:
        """Check if content violates OpenAI's content policy.
        
        Args:
            text: The text to moderate
            
        Returns:
            Dictionary with moderation results
        """
        try:
            response = self.client.moderations.create(input=text)
            result = response.results[0]
            
            # Convert to a more usable dictionary
            moderation_result = {
                "flagged": result.flagged,
                "categories": {}
            }
            
            # Add categories that were flagged
            for category, flagged in result.categories.model_dump().items():
                if flagged:
                    score = getattr(result.category_scores, category)
                    moderation_result["categories"][category] = float(score)
            
            return moderation_result
            
        except Exception as e:
            self.logger.error(f"Error moderating content: {e}")
            # Return a safe default
            return {"flagged": False, "categories": {}}
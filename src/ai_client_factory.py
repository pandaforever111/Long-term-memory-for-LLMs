#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AI Client Factory Module

This module provides a factory for creating AI clients (OpenAI or Puter.js)
based on configuration settings.
"""

import logging
from typing import Optional, Union

from src.config import Config

# Import clients
try:
    from src.openai_client import OpenAIClient
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from src.puter_client import PuterClient
    PUTER_AVAILABLE = True
except ImportError:
    PUTER_AVAILABLE = False


class AIClientFactory:
    """Factory for creating AI clients based on configuration."""
    
    @staticmethod
    def create_client(config: Config) -> Union["OpenAIClient", "PuterClient"]:
        """Create and return an AI client based on configuration.
        
        Args:
            config: Configuration object with AI provider settings
            
        Returns:
            An instance of OpenAIClient or PuterClient
            
        Raises:
            ImportError: If the requested client is not available
            ValueError: If the AI provider is not supported
        """
        logger = logging.getLogger("memory_agent.ai_client_factory")
        
        # Get the AI provider from config
        ai_provider = getattr(config, "ai_provider", "openai").lower()
        
        if ai_provider == "openai":
            if not OPENAI_AVAILABLE:
                logger.error("OpenAI package is not installed. Please install it with 'pip install openai'.")
                raise ImportError("OpenAI package is required but not installed")
            
            logger.info("Creating OpenAI client")
            return OpenAIClient(config)
            
        elif ai_provider == "puter" or ai_provider == "puter.js":
            if not PUTER_AVAILABLE:
                logger.error("Puter.js client module not found.")
                raise ImportError("Puter.js client module is required but not available")
            
            logger.info("Creating Puter.js client")
            return PuterClient(config)
            
        else:
            logger.error(f"Unsupported AI provider: {ai_provider}")
            raise ValueError(f"Unsupported AI provider: {ai_provider}. Supported providers: openai, puter")
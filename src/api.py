#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
API Interface for GPT Memory Agent

This module provides a FastAPI interface for the GPT Memory Agent.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Union, Any
from datetime import datetime

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.memory_agent import MemoryAgent
from src.config import Config


# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("memory_agent.api")

# Initialize the FastAPI app
app = FastAPI(
    title="GPT Memory Agent API",
    description="API for interacting with the GPT Memory Agent",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global memory agent instance
memory_agent = None


# Pydantic models for request/response validation
class MessageRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")
    message: str = Field(..., description="The user's message")
    conversation_id: Optional[str] = Field(None, description="Optional conversation identifier")


class MemoryRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")
    query: str = Field(..., description="Query to search for relevant memories")
    limit: int = Field(5, description="Maximum number of memories to retrieve")


class DeleteMemoryRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")
    content_pattern: str = Field(..., description="Pattern to match against memory content")


class StatsRequest(BaseModel):
    user_id: str = Field(..., description="Unique identifier for the user")


# Dependency to get the memory agent instance
def get_memory_agent():
    global memory_agent
    if memory_agent is None:
        # Initialize with default config or from environment
        config_path = os.environ.get("MEMORY_AGENT_CONFIG")
        memory_agent = MemoryAgent(config_path)
    return memory_agent


@app.on_event("startup")
async def startup_event():
    """Initialize resources on startup."""
    logger.info("Starting GPT Memory Agent API")
    # Initialize the memory agent
    get_memory_agent()


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown."""
    logger.info("Shutting down GPT Memory Agent API")
    # Any cleanup needed


@app.get("/")
async def root():
    """Root endpoint returning API information."""
    return {
        "name": "GPT Memory Agent API",
        "version": "0.1.0",
        "status": "running"
    }


@app.post("/generate")
async def generate_response(request: MessageRequest, agent: MemoryAgent = Depends(get_memory_agent)):
    """Generate a response with memory context."""
    try:
        result = agent.generate_response(
            user_id=request.user_id,
            message=request.message,
            conversation_id=request.conversation_id
        )
        return result
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process")
async def process_message(request: MessageRequest, agent: MemoryAgent = Depends(get_memory_agent)):
    """Process a message to extract and store memories without generating a response."""
    try:
        result = agent.process_message(
            user_id=request.user_id,
            message=request.message,
            conversation_id=request.conversation_id
        )
        return result
    except Exception as e:
        logger.error(f"Error processing message: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memories")
async def retrieve_memories(request: MemoryRequest, agent: MemoryAgent = Depends(get_memory_agent)):
    """Retrieve relevant memories for a user based on a query."""
    try:
        memories = agent.retrieve_memories(
            user_id=request.user_id,
            query=request.query,
            limit=request.limit
        )
        return {"memories": memories}
    except Exception as e:
        logger.error(f"Error retrieving memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memories/delete")
async def delete_memories(
    request: DeleteMemoryRequest, 
    agent: MemoryAgent = Depends(get_memory_agent)
):
    """Delete memories matching a content pattern."""
    try:
        deleted_ids = agent.memory_store.delete_memories_by_content(
            user_id=request.user_id,
            content_pattern=request.content_pattern
        )
        return {"deleted_memory_ids": deleted_ids}
    except Exception as e:
        logger.error(f"Error deleting memories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/stats")
async def get_stats(request: StatsRequest, agent: MemoryAgent = Depends(get_memory_agent)):
    """Get memory statistics for a user."""
    try:
        stats = agent.memory_store.get_user_memory_stats(request.user_id)
        return stats
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


def start_api(host: str = "0.0.0.0", port: int = 8000, config_path: Optional[str] = None):
    """Start the API server.
    
    Args:
        host: Host to bind the server to
        port: Port to bind the server to
        config_path: Path to configuration file
    """
    import uvicorn
    
    # Set config path in environment if provided
    if config_path:
        os.environ["MEMORY_AGENT_CONFIG"] = config_path
    
    # Start the server
    uvicorn.run("src.api:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="GPT Memory Agent API")
    parser.add_argument(
        "--host", 
        type=str, 
        default="0.0.0.0",
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port to bind the server to"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Path to configuration file"
    )
    args = parser.parse_args()
    
    start_api(args.host, args.port, args.config)
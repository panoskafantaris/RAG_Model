"""
Pydantic models for API request/response validation.
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class NewChatRequest(BaseModel):
    """Request to create a new chat."""
    title: Optional[str] = Field(default="Νέα Συνομιλία", max_length=200)


class MessageRequest(BaseModel):
    """Request to send a message."""
    role: str = Field(default="user", pattern="^(user|assistant|system)$")
    content: str = Field(..., min_length=1, max_length=10000)
    
    @validator('content')
    def content_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty or whitespace only')
        return v.strip()


class ChatSummary(BaseModel):
    """Summary of a chat for listing."""
    id: str
    title: str
    last_updated: str
    message_count: int


class Message(BaseModel):
    """Single message in a chat."""
    role: str
    content: str
    timestamp: str


class ChatDetail(BaseModel):
    """Full chat details with message history."""
    id: str
    title: str
    messages: List[Message]
    created: str
    updated: str


class SourceDocument(BaseModel):
    """Information about a retrieved source document."""
    content: str = Field(..., description="Document content snippet")
    source: str = Field(..., description="Source filename")
    relevance_score: float = Field(..., ge=0, le=1, description="Similarity score")


class ChatResponse(BaseModel):
    """Response from the chat endpoint."""
    answer: str
    sources: List[SourceDocument]
    metadata: Optional[Dict[str, Any]] = None


class UploadResponse(BaseModel):
    """Response after file upload."""
    filename: str
    status: str
    message: Optional[str] = None


class IngestionResponse(BaseModel):
    """Response after ingestion process."""
    success: bool
    message: str
    documents_loaded: int
    chunks_created: int


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    services: Dict[str, Any]  # Changed from Dict[str, bool] to allow nested GPU info


class StatsResponse(BaseModel):
    """System statistics."""
    total_chats: int
    total_messages: int
    avg_messages_per_chat: float
    vector_store_initialized: bool
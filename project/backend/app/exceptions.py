"""
Custom exceptions for better error handling and user feedback.
"""

class RAGException(Exception):
    """Base exception for RAG system."""
    pass

class ChatNotFoundException(RAGException):
    """Raised when a chat ID doesn't exist."""
    def __init__(self, chat_id: str):
        self.chat_id = chat_id
        super().__init__(f"Chat with ID '{chat_id}' not found")

class VectorStoreNotInitializedException(RAGException):
    """Raised when vector store index doesn't exist."""
    def __init__(self):
        super().__init__(
            "Vector store not initialized. Please run ingestion first."
        )

class ModelLoadException(RAGException):
    """Raised when model fails to load."""
    def __init__(self, model_name: str, original_error: Exception):
        self.model_name = model_name
        self.original_error = original_error
        super().__init__(
            f"Failed to load model '{model_name}': {str(original_error)}"
        )

class IngestionException(RAGException):
    """Raised when document ingestion fails."""
    def __init__(self, message: str):
        super().__init__(f"Ingestion failed: {message}")
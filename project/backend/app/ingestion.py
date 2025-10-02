"""
Complete document ingestion pipeline.
Handles file upload, text extraction, chunking, and indexing.
"""
import os
from pathlib import Path
from typing import List, Optional
from fastapi import UploadFile

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from .config import KNOWLEDGE_DIR, RAG_CONFIG
from .vectorstore import add_documents, reset_vectorstore
from .exceptions import IngestionException
from .logger import setup_logger

logger = setup_logger(__name__)


async def save_upload(file: UploadFile) -> str:
    """
    Save uploaded file to knowledge directory.
    
    Args:
        file: Uploaded file
    
    Returns:
        Path to saved file
    """
    try:
        # Ensure directory exists
        KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create safe filename
        filename = file.filename.replace("/", "_").replace("\\", "_")
        filepath = KNOWLEDGE_DIR / filename
        
        # Save file
        content = await file.read()
        
        with open(filepath, "wb") as f:
            f.write(content)
        
        logger.info(f"Saved uploaded file: {filepath}")
        return str(filepath)
    
    except Exception as e:
        logger.error(f"Failed to save upload: {e}")
        raise IngestionException(f"Could not save file: {e}")


def load_text_file(filepath: Path) -> str:
    """
    Load text from a file with encoding fallback.
    
    Args:
        filepath: Path to file
    
    Returns:
        File content as string
    """
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
    
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                content = f.read()
            logger.debug(f"Successfully read {filepath} with {encoding} encoding")
            return content
        except UnicodeDecodeError:
            continue
    
    raise IngestionException(f"Could not decode file {filepath} with any encoding")


def load_documents_from_directory(directory: Path) -> List[Document]:
    """
    Load all text documents from a directory.
    
    Args:
        directory: Directory path
    
    Returns:
        List of LangChain Documents
    """
    documents = []
    
    # Supported file extensions
    supported_extensions = {'.txt', '.md', '.text'}
    
    if not directory.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return documents
    
    for filepath in directory.rglob('*'):
        if filepath.is_file() and filepath.suffix.lower() in supported_extensions:
            try:
                content = load_text_file(filepath)
                
                # Create document with metadata
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": filepath.name,
                        "filepath": str(filepath),
                        "file_type": filepath.suffix,
                    }
                )
                documents.append(doc)
                logger.info(f"Loaded document: {filepath.name}")
            
            except Exception as e:
                logger.error(f"Failed to load {filepath}: {e}")
    
    logger.info(f"Loaded {len(documents)} documents from {directory}")
    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into smaller chunks for better retrieval.
    
    Args:
        documents: List of documents
    
    Returns:
        List of chunked documents
    """
    if not documents:
        return []
    
    # Create text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=RAG_CONFIG["chunk_size"],
        chunk_overlap=RAG_CONFIG["chunk_overlap"],
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Split documents
    chunked_docs = text_splitter.split_documents(documents)
    
    # Add chunk metadata
    for i, doc in enumerate(chunked_docs):
        doc.metadata["chunk_id"] = i
    
    logger.info(f"Split {len(documents)} documents into {len(chunked_docs)} chunks")
    return chunked_docs


def ingest_directory(directory: Path = KNOWLEDGE_DIR, rebuild: bool = False) -> dict:
    """
    Ingest all documents from a directory into the vector store.
    
    Args:
        directory: Directory containing documents
        rebuild: If True, rebuild index from scratch
    
    Returns:
        Dictionary with ingestion statistics
    """
    try:
        logger.info(f"Starting ingestion from {directory} (rebuild={rebuild})")
        
        # Reset vector store if rebuilding
        if rebuild:
            logger.info("Rebuilding vector store from scratch")
            reset_vectorstore()
        
        # Load documents
        documents = load_documents_from_directory(directory)
        
        if not documents:
            logger.warning("No documents found to ingest")
            return {
                "success": False,
                "message": "No documents found",
                "documents_loaded": 0,
                "chunks_created": 0
            }
        
        # Chunk documents
        chunked_docs = chunk_documents(documents)
        
        # Add to vector store
        success = add_documents(chunked_docs)
        
        if success:
            logger.info("Ingestion completed successfully")
            return {
                "success": True,
                "message": "Ingestion completed successfully",
                "documents_loaded": len(documents),
                "chunks_created": len(chunked_docs)
            }
        else:
            raise IngestionException("Failed to add documents to vector store")
    
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        return {
            "success": False,
            "message": str(e),
            "documents_loaded": 0,
            "chunks_created": 0
        }


async def ingest_single_file(filepath: str) -> dict:
    """
    Ingest a single file into the vector store.
    
    Args:
        filepath: Path to the file
    
    Returns:
        Dictionary with ingestion result
    """
    try:
        path = Path(filepath)
        
        if not path.exists():
            raise IngestionException(f"File not found: {filepath}")
        
        # Load single document
        content = load_text_file(path)
        doc = Document(
            page_content=content,
            metadata={
                "source": path.name,
                "filepath": str(path),
                "file_type": path.suffix,
            }
        )
        
        # Chunk document
        chunked_docs = chunk_documents([doc])
        
        # Add to vector store
        success = add_documents(chunked_docs)
        
        if success:
            return {
                "success": True,
                "message": f"File {path.name} ingested successfully",
                "chunks_created": len(chunked_docs)
            }
        else:
            raise IngestionException("Failed to add document to vector store")
    
    except Exception as e:
        logger.error(f"Failed to ingest file {filepath}: {e}")
        return {
            "success": False,
            "message": str(e),
            "chunks_created": 0
        }
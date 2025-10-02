"""
Configuration management for the RAG system.
Centralizes all settings for easy modification and environment-specific configs.
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
KNOWLEDGE_DIR = DATA_DIR / "knowledge"
INSTRUCTIONS_DIR = DATA_DIR / "instructions"
INDEX_DIR = BASE_DIR / "faiss_index"

# Ensure directories exist
KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
INSTRUCTIONS_DIR.mkdir(parents=True, exist_ok=True)

# Model configurations
LLM_MODEL_NAME = "ilsp/Llama-Krikri-8B-Instruct"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# LLM Generation parameters
LLM_CONFIG = {
    "max_new_tokens": 512,
    "temperature": 0.7,
    "top_p": 0.9,
    "do_sample": True,
    "repetition_penalty": 1.1,
}

# RAG parameters
RAG_CONFIG = {
    "top_k": 3,  # Number of documents to retrieve
    "chunk_size": 500,  # Characters per chunk
    "chunk_overlap": 50,  # Overlap between chunks
    "min_relevance_score": 0.3,  # Minimum similarity score (0-1)
}

# CORS settings
CORS_ORIGINS = [
    "http://localhost:4200",
    "http://127.0.0.1:4200",
]

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# System instructions (loaded from file or default)
def load_system_instructions():
    """Load system instructions from files."""
    persona_file = INSTRUCTIONS_DIR / "persona.txt"
    rules_file = INSTRUCTIONS_DIR / "rules.txt"
    
    persona = ""
    rules = ""
    
    if persona_file.exists():
        persona = persona_file.read_text(encoding="utf-8")
    
    if rules_file.exists():
        rules = rules_file.read_text(encoding="utf-8")
    
    return f"{persona}\n\n{rules}".strip()

SYSTEM_INSTRUCTION = load_system_instructions() or """
Είσαι ένας βοηθός πληροφορικής για το εσωτερικό τμήμα ΙΤ της εταιρείας.
Απαντάς πάντα στα ελληνικά, εκτός αν ζητηθεί διαφορετικά.
Οι απαντήσεις σου είναι σύντομες, κατανοητές και πρακτικές.
Αν δεν γνωρίζεις κάτι, το δηλώνεις ξεκάθαρα.
"""
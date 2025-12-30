"""
Configuration settings for the Multi-Course RAG Student Assistant.
"""
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_PERSIST_DIR = DATA_DIR / "chroma_db"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ChromaDB settings
CHROMA_COLLECTION_NAME = "course_documents"

# Chunking settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval settings
TOP_K_RESULTS = 5
SIMILARITY_THRESHOLD = 0.3

# Embedding settings
# EMBEDDING_PROVIDER: "local" (free, offline) or "openai" (paid, better quality)
EMBEDDING_PROVIDER = "openai"

# Local embedding model (used when EMBEDDING_PROVIDER = "local")
# Uses sentence-transformers, runs on CPU, no API key needed
LOCAL_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# OpenAI embedding model (used when EMBEDDING_PROVIDER = "openai")
# Options: "text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"

# LLM settings (using OpenAI - can be swapped for local models)
# Set OPENAI_API_KEY in environment or .env file
LLM_MODEL = "gpt-4o-mini"
LLM_TEMPERATURE = 0.1
LLM_MAX_TOKENS = 1500

# Document types
DOCUMENT_TYPES = [
    "lecture",
    "assignment",
    "syllabus",
    "exam",
    "schedule",
    "misc"
]

# Special course IDs
MISC_COURSE_ID = "MISC"
AUTO_COURSE_ID = "AUTO"

# UI settings
MAX_CHAT_HISTORY = 10  # Number of recent turns to include in context

# Logging
LOG_FILE = LOGS_DIR / "rag_assistant.log"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


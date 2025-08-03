"""HIPAA QA System - A RAG-based chatbot for HIPAA regulations."""

__version__ = "0.1.0"
__author__ = "HIPAA QA Team"
__email__ = "team@example.com"
__description__ = "HIPAA Regulation QA System using RAG with OpenAI and pgvector"

from typing import Final

# Package constants
PACKAGE_NAME: Final[str] = "hipaa-qa-system"
OPENAI_EMBEDDING_MODEL: Final[str] = "text-embedding-3-large"
OPENAI_CHAT_MODEL: Final[str] = "gpt-4"
EMBEDDING_DIMENSION: Final[int] = 3072  # text-embedding-3-large dimension
MAX_TOKENS_PER_CHUNK: Final[int] = 500
SIMILARITY_THRESHOLD: Final[float] = 0.7
DEFAULT_TOP_K: Final[int] = 5

__all__ = [
    "__version__",
    "__author__", 
    "__email__",
    "__description__",
    "PACKAGE_NAME",
    "OPENAI_EMBEDDING_MODEL",
    "OPENAI_CHAT_MODEL", 
    "EMBEDDING_DIMENSION",
    "MAX_TOKENS_PER_CHUNK",
    "SIMILARITY_THRESHOLD",
    "DEFAULT_TOP_K",
]
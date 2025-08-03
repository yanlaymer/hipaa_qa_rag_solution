"""API module for HIPAA QA System."""

from .main import create_app, get_app
from .routes import health, ingestion, qa

__all__ = [
    "create_app",
    "get_app",
    "health",
    "ingestion", 
    "qa",
]
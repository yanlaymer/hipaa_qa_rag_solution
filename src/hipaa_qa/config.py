"""Configuration management for HIPAA QA System."""

from functools import lru_cache
from typing import Optional

from pydantic import Field, validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with validation and environment variable support."""
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., description="OpenAI API key for embeddings and chat")
    openai_embedding_model: str = Field(
        default="text-embedding-3-large",
        description="OpenAI embedding model to use"
    )
    openai_chat_model: str = Field(
        default="gpt-4",
        description="OpenAI chat model for question answering"
    )
    openai_max_retries: int = Field(default=3, description="Max retries for OpenAI API calls")
    openai_timeout: int = Field(default=30, description="Timeout for OpenAI API calls in seconds")
    
    # Database Configuration
    db_host: str = Field(default="localhost", description="PostgreSQL host", alias="DB_HOST")
    db_port: int = Field(default=5432, description="PostgreSQL port", alias="DB_PORT")
    db_name: str = Field(default="hipaa_qa", description="Database name", alias="POSTGRES_DB")
    db_user: str = Field(default="postgres", description="Database user", alias="POSTGRES_USER")
    db_password: str = Field(default="", description="Database password", alias="POSTGRES_PASSWORD")
    db_pool_size: int = Field(default=10, description="Database connection pool size")
    db_max_overflow: int = Field(default=20, description="Database connection pool max overflow")
    
    # Vector Search Configuration
    embedding_dimension: int = Field(
        default=3072, 
        description="Embedding vector dimension for text-embedding-3-large"
    )
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for relevant chunks"
    )
    max_chunks_retrieved: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of chunks to retrieve for context"
    )
    chunk_overlap_tokens: int = Field(
        default=50,
        description="Number of overlapping tokens between chunks"
    )
    
    # API Configuration
    api_host: str = Field(default="0.0.0.0", description="FastAPI host")
    api_port: int = Field(default=8000, description="FastAPI port", alias="API_PORT")
    api_workers: int = Field(default=1, description="Number of uvicorn workers")
    api_debug: bool = Field(default=False, description="Enable debug mode", alias="DEBUG")
    api_cors_origins: list[str] = Field(
        default=["*"],
        description="Allowed CORS origins"
    )
    
    # Frontend Configuration
    gradio_host: str = Field(default="0.0.0.0", description="Gradio host")
    gradio_port: int = Field(default=7860, description="Gradio port", alias="FRONTEND_PORT")
    gradio_share: bool = Field(default=False, description="Enable Gradio sharing")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", description="Logging level")
    log_format: str = Field(
        default="json",
        description="Log format: 'json' or 'text'"
    )
    
    # Data Paths
    data_path: str = Field(default="data", description="Base data directory path")
    chunks_file: str = Field(
        default="data/clean/hipaa_rag_chunks.json",
        description="Path to processed chunks file"
    )
    
    # Performance Configuration
    batch_size: int = Field(default=100, description="Batch size for embedding operations")
    request_timeout: int = Field(default=60, description="Request timeout in seconds")
    
    @validator("openai_api_key", pre=True)
    def validate_openai_key(cls, v: str) -> str:
        """Validate OpenAI API key format."""
        if not v.startswith("sk-"):
            raise ValueError("OpenAI API key must start with 'sk-'")
        return v
        
    @validator("log_level", pre=True)
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
        
    @validator("log_format", pre=True)
    def validate_log_format(cls, v: str) -> str:
        """Validate log format."""
        valid_formats = ["json", "text"]
        if v.lower() not in valid_formats:
            raise ValueError(f"Log format must be one of: {valid_formats}")
        return v.lower()

    @property
    def database_url(self) -> str:
        """Get the complete database URL."""
        auth_part = f"{self.db_user}"
        if self.db_password:
            auth_part += f":{self.db_password}"
        return (
            f"postgresql+asyncpg://{auth_part}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )
        
    @property
    def sync_database_url(self) -> str:
        """Get the synchronous database URL for migrations."""
        auth_part = f"{self.db_user}"
        if self.db_password:
            auth_part += f":{self.db_password}"
        return (
            f"postgresql://{auth_part}"
            f"@{self.db_host}:{self.db_port}/{self.db_name}"
        )

    class Config:
        """Pydantic model configuration."""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        env_prefix = ""
        extra = "ignore"  # Allow extra env vars to be ignored


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()


# Global settings instance
settings = get_settings()
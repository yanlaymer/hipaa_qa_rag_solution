"""SQLAlchemy models for HIPAA QA database tables."""

from datetime import datetime
from typing import List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Mapped, mapped_column

# Create declarative base
Base = declarative_base()


class DocumentChunkTable(Base):
    """Database model for storing document chunks with embeddings."""
    
    __tablename__ = "document_chunks"
    
    # Primary key
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Chunk identification
    chunk_id: Mapped[int] = mapped_column(Integer, unique=True, nullable=False, index=True)
    
    # Content
    content: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Section information (indexed for fast filtering)
    section_id: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    section_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    section_title: Mapped[str] = mapped_column(String(500), nullable=False)
    full_reference: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    cfr_citation: Mapped[Optional[str]] = mapped_column(String(300), nullable=True, index=True)
    parent_section: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    hierarchy_level: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    
    # Chunk metadata
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    total_chunks: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_size: Mapped[int] = mapped_column(Integer, nullable=False)
    word_count: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Content classification (indexed for filtering)
    contains_definitions: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    contains_penalties: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    contains_requirements: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    
    # Content type and compliance level (indexed for filtering)
    content_type: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    compliance_level: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    
    # References and terms (stored as JSON for flexibility)
    references: Mapped[List[str]] = mapped_column("references_json", JSONB, default=list)
    key_terms: Mapped[List[str]] = mapped_column("key_terms_json", JSONB, default=list)
    
    # Vector embedding (pgvector column)
    embedding: Mapped[Optional[List[float]]] = mapped_column(
        Vector(3072),  # text-embedding-3-large dimension
        nullable=True,
        comment="Vector embedding for semantic search"
    )
    
    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )
    
    # Optional metadata (stored as JSON for extensibility)
    metadata_json: Mapped[Optional[dict]] = mapped_column(
        JSONB,
        nullable=True,
        comment="Additional metadata in JSON format"
    )
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"<DocumentChunk(id={self.id}, chunk_id={self.chunk_id}, "
            f"section_id='{self.section_id}', content_length={len(self.content)})>"
        )
        
    def to_dict(self) -> dict:
        """Convert model to dictionary representation."""
        return {
            "id": self.id,
            "chunk_id": self.chunk_id,
            "content": self.content,
            "section_id": self.section_id,
            "section_type": self.section_type,
            "section_title": self.section_title,
            "full_reference": self.full_reference,
            "cfr_citation": self.cfr_citation,
            "parent_section": self.parent_section,
            "hierarchy_level": self.hierarchy_level,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "chunk_size": self.chunk_size,
            "word_count": self.word_count,
            "contains_definitions": self.contains_definitions,
            "contains_penalties": self.contains_penalties,
            "contains_requirements": self.contains_requirements,
            "content_type": self.content_type,
            "compliance_level": self.compliance_level,
            "references": self.references,
            "key_terms": self.key_terms,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata_json": self.metadata_json,
        }


class QueryLog(Base):
    """Model for logging user queries and responses for analytics."""
    
    __tablename__ = "query_logs"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    
    # Query information
    question: Mapped[str] = mapped_column(Text, nullable=False)
    answer: Mapped[str] = mapped_column(Text, nullable=False)
    
    # Performance metrics
    processing_time_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    chunks_retrieved: Mapped[int] = mapped_column(Integer, nullable=False)
    
    # Quality metrics
    confidence_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    similarity_threshold_used: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Model information
    embedding_model: Mapped[str] = mapped_column(String(100), nullable=False)
    chat_model: Mapped[str] = mapped_column(String(100), nullable=False)
    
    # Retrieved chunk IDs (for analysis)
    retrieved_chunk_ids: Mapped[List[int]] = mapped_column(JSONB, default=list)
    
    # User session information (optional)
    session_id: Mapped[Optional[str]] = mapped_column(String(100), nullable=True, index=True)
    user_ip: Mapped[Optional[str]] = mapped_column(String(45), nullable=True)  # IPv6 compatible
    
    # Timestamp
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True
    )
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return (
            f"<QueryLog(id={self.id}, question_length={len(self.question)}, "
            f"processing_time={self.processing_time_ms}ms)>"
        )
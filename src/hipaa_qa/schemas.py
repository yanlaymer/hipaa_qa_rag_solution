"""Pydantic schemas for HIPAA QA System API contracts."""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class ContentType(str, Enum):
    """Types of regulatory content."""
    DEFINITION = "definition"
    REQUIREMENT = "requirement"
    GENERAL = "general"
    PENALTY = "penalty"
    PROCEDURE = "procedure"


class ComplianceLevel(str, Enum):
    """Compliance level indicators."""
    MANDATORY = "mandatory"
    REQUIRED = "required"
    PERMITTED = "permitted"
    PROHIBITED = "prohibited"
    INFORMATIONAL = "informational"


class SectionType(str, Enum):
    """HIPAA regulation section types."""
    PART = "part"
    SUBPART = "subpart"
    SECTION = "section"
    SUBSECTION = "subsection"
    PARAGRAPH = "paragraph"


class ChunkMetadata(BaseModel):
    """Metadata for a regulation text chunk."""
    
    section_id: str = Field(..., description="Unique section identifier")
    section_type: SectionType = Field(..., description="Type of regulation section")
    section_title: str = Field(..., description="Title of the section")
    full_reference: str = Field(..., description="Complete CFR reference")
    cfr_citation: Optional[str] = Field(None, description="Specific CFR citation")
    parent_section: Optional[str] = Field(None, description="Parent section ID")
    hierarchy_level: int = Field(..., ge=1, le=10, description="Hierarchy depth level")
    
    # Chunk-specific metadata
    chunk_index: int = Field(..., ge=0, description="Index of this chunk within section")
    total_chunks: int = Field(..., ge=1, description="Total chunks for this section")
    chunk_size: int = Field(..., ge=0, description="Character count of chunk")
    word_count: int = Field(..., ge=0, description="Word count of chunk")
    
    # Content classification
    contains_definitions: bool = Field(default=False, description="Contains regulatory definitions")
    contains_penalties: bool = Field(default=False, description="Contains penalty information")
    contains_requirements: bool = Field(default=False, description="Contains compliance requirements")
    
    # Reference tracking
    references: List[str] = Field(default_factory=list, description="Referenced sections/citations")
    key_terms: List[str] = Field(default_factory=list, description="Important regulatory terms")
    
    # Classification
    content_type: ContentType = Field(default=ContentType.GENERAL, description="Type of content")
    compliance_level: ComplianceLevel = Field(
        default=ComplianceLevel.INFORMATIONAL,
        description="Compliance importance level"
    )


class DocumentChunk(BaseModel):
    """A chunk of regulatory text with metadata and embeddings."""
    
    chunk_id: int = Field(..., ge=0, description="Unique chunk identifier")
    content: str = Field(..., min_length=1, description="The actual regulation text")
    metadata: ChunkMetadata = Field(..., description="Chunk metadata and classification")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding of the content")
    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default_factory=datetime.utcnow)
    
    @validator("embedding")
    def validate_embedding_dimension(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        """Validate embedding dimension matches expected size."""
        if v is not None and len(v) != 3072:  # text-embedding-3-large dimension
            raise ValueError(f"Embedding must have 3072 dimensions, got {len(v)}")
        return v
        
    @validator("content")
    def validate_content_not_empty(cls, v: str) -> str:
        """Ensure content is not just whitespace."""
        if not v.strip():
            raise ValueError("Content cannot be empty or only whitespace")
        return v.strip()


class QuestionRequest(BaseModel):
    """Request schema for asking a question."""
    
    question: str = Field(..., min_length=5, max_length=500, description="User's question about HIPAA")
    max_chunks: Optional[int] = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of context chunks to retrieve"
    )
    similarity_threshold: Optional[float] = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold for chunk relevance"
    )
    include_metadata: bool = Field(
        default=False,
        description="Include chunk metadata in response"
    )
    
    @validator("question")
    def validate_question_content(cls, v: str) -> str:
        """Validate question is meaningful."""
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Question cannot be empty")
        if len(cleaned.split()) < 2:
            raise ValueError("Question must contain at least 2 words")
        return cleaned


class SourceReference(BaseModel):
    """Reference to a source regulation section."""
    
    section_id: str = Field(..., description="Section identifier")
    cfr_citation: str = Field(..., description="CFR citation")
    section_title: str = Field(..., description="Section title")
    content_excerpt: str = Field(..., description="Relevant excerpt from the section")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    chunk_id: int = Field(..., description="Source chunk ID")


class AnswerResponse(BaseModel):
    """Response schema for question answers."""
    
    question: str = Field(..., description="Original question")
    answer: str = Field(..., description="Generated answer with citations")
    sources: List[SourceReference] = Field(
        default_factory=list,
        description="Source references used to generate the answer"
    )
    confidence_score: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Confidence in the answer quality"
    )
    processing_time_ms: Optional[int] = Field(
        None,
        ge=0,
        description="Time taken to process the request in milliseconds"
    )
    model_used: str = Field(..., description="OpenAI model used for generation")
    chunks_retrieved: int = Field(..., ge=0, description="Number of chunks retrieved")
    
    # Optional metadata
    metadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Additional metadata about the response"
    )


class HealthResponse(BaseModel):
    """Health check response schema."""
    
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(..., description="Application version")
    database_connected: bool = Field(..., description="Database connection status")
    openai_accessible: bool = Field(..., description="OpenAI API accessibility")
    chunks_indexed: int = Field(..., ge=0, description="Number of chunks in database")


class ErrorResponse(BaseModel):
    """Error response schema."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    error_code: Optional[str] = Field(None, description="Application-specific error code")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = Field(None, description="Request tracking ID")


class BulkIngestionRequest(BaseModel):
    """Request schema for bulk document ingestion."""
    
    source_file: str = Field(..., description="Path to source chunks file")
    batch_size: int = Field(default=100, ge=1, le=1000, description="Batch size for processing")
    overwrite_existing: bool = Field(default=False, description="Whether to overwrite existing data")
    validate_embeddings: bool = Field(default=True, description="Whether to validate embedding quality")


class IngestionResponse(BaseModel):
    """Response schema for document ingestion."""
    
    status: str = Field(..., description="Ingestion status")
    chunks_processed: int = Field(..., ge=0, description="Number of chunks processed")
    chunks_inserted: int = Field(..., ge=0, description="Number of chunks successfully inserted")
    chunks_failed: int = Field(..., ge=0, description="Number of chunks that failed")
    processing_time_ms: int = Field(..., ge=0, description="Total processing time")
    errors: List[str] = Field(default_factory=list, description="Any errors encountered")


class SearchRequest(BaseModel):
    """Request schema for semantic search."""
    
    query: str = Field(..., min_length=1, description="Search query")
    max_results: int = Field(default=10, ge=1, le=50, description="Maximum results to return")
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold"
    )
    filter_content_types: Optional[List[ContentType]] = Field(
        None,
        description="Filter by content types"
    )
    filter_sections: Optional[List[str]] = Field(
        None,
        description="Filter by specific sections"
    )


class SearchResult(BaseModel):
    """Individual search result."""
    
    chunk_id: int = Field(..., description="Chunk identifier")
    content: str = Field(..., description="Chunk content")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Similarity score")
    metadata: ChunkMetadata = Field(..., description="Chunk metadata")


class SearchResponse(BaseModel):
    """Response schema for semantic search."""
    
    query: str = Field(..., description="Original search query")
    results: List[SearchResult] = Field(default_factory=list, description="Search results")
    total_results: int = Field(..., ge=0, description="Total number of results found")
    search_time_ms: int = Field(..., ge=0, description="Search execution time")
    similarity_threshold_used: float = Field(..., description="Similarity threshold applied")
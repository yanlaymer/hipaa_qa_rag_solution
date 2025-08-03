"""Service for ingesting and processing HIPAA document chunks."""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from loguru import logger

from ..database import ChunkRepository, DatabaseManager
from ..schemas import ChunkMetadata, ComplianceLevel, ContentType, DocumentChunk, SectionType
from .embedding_service import EmbeddingService


class IngestionService:
    """Service for ingesting document chunks into the database."""
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        embedding_service: EmbeddingService,
    ) -> None:
        """Initialize ingestion service."""
        self.db_manager = db_manager
        self.embedding_service = embedding_service
        self.repository = ChunkRepository(db_manager)
        
    async def ingest_from_json(
        self,
        json_file_path: str,
        batch_size: int = 100,
        overwrite_existing: bool = False,
        generate_embeddings: bool = True,
    ) -> Tuple[int, int, List[str]]:
        """
        Ingest document chunks from JSON file.
        
        Args:
            json_file_path: Path to the chunks JSON file
            batch_size: Number of chunks to process per batch
            overwrite_existing: Whether to overwrite existing data
            generate_embeddings: Whether to generate embeddings
            
        Returns:
            Tuple of (success_count, error_count, error_messages)
        """
        logger.info(f"Starting ingestion from {json_file_path}")
        
        # Validate file exists
        json_path = Path(json_file_path)
        if not json_path.exists():
            raise FileNotFoundError(f"Chunks file not found: {json_file_path}")
            
        # Clear existing data if requested
        if overwrite_existing:
            deleted_count = await self.repository.delete_all_chunks()
            logger.info(f"Cleared {deleted_count} existing chunks")
            
        # Load and parse JSON data
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                raw_chunks = json.load(f)
        except Exception as e:
            raise ValueError(f"Failed to load JSON file: {e}")
            
        logger.info(f"Loaded {len(raw_chunks)} chunks from JSON")
        
        # Convert to DocumentChunk objects
        chunks = []
        conversion_errors = []
        
        for i, raw_chunk in enumerate(raw_chunks):
            try:
                chunk = self._convert_raw_chunk(raw_chunk)
                chunks.append(chunk)
            except Exception as e:
                error_msg = f"Failed to convert chunk {i}: {e}"
                conversion_errors.append(error_msg)
                logger.warning(error_msg)
                
        logger.info(f"Converted {len(chunks)} chunks, {len(conversion_errors)} conversion errors")
        
        # Generate embeddings if requested
        if generate_embeddings:
            logger.info("Generating embeddings for chunks...")
            await self._add_embeddings_to_chunks(chunks, batch_size)
            
        # Insert chunks into database
        success_count, error_count = await self.repository.bulk_insert_chunks(
            chunks, batch_size
        )
        
        all_errors = conversion_errors
        if error_count > 0:
            all_errors.append(f"{error_count} chunks failed database insertion")
            
        logger.info(
            f"Ingestion completed: {success_count} successful, "
            f"{error_count + len(conversion_errors)} errors"
        )
        
        return success_count, error_count + len(conversion_errors), all_errors
        
    def _convert_raw_chunk(self, raw_chunk: dict) -> DocumentChunk:
        """Convert raw JSON chunk data to DocumentChunk object."""
        try:
            # Extract metadata
            raw_metadata = raw_chunk.get("metadata", {})
            
            # Map section type
            section_type_map = {
                "part": SectionType.PART,
                "subpart": SectionType.SUBPART, 
                "section": SectionType.SECTION,
                "subsection": SectionType.SUBSECTION,
                "paragraph": SectionType.PARAGRAPH,
            }
            section_type = section_type_map.get(
                raw_metadata.get("section_type", "section").lower(),
                SectionType.SECTION
            )
            
            # Map content type
            content_type_map = {
                "definition": ContentType.DEFINITION,
                "requirement": ContentType.REQUIREMENT,
                "general": ContentType.GENERAL,
                "penalty": ContentType.PENALTY,
                "procedure": ContentType.PROCEDURE,
            }
            content_type = content_type_map.get(
                raw_metadata.get("content_type", "general").lower(),
                ContentType.GENERAL
            )
            
            # Map compliance level
            compliance_level_map = {
                "mandatory": ComplianceLevel.MANDATORY,
                "required": ComplianceLevel.REQUIRED,
                "permitted": ComplianceLevel.PERMITTED,
                "prohibited": ComplianceLevel.PROHIBITED,
                "informational": ComplianceLevel.INFORMATIONAL,
            }
            compliance_level = compliance_level_map.get(
                raw_metadata.get("compliance_level", "informational").lower(),
                ComplianceLevel.INFORMATIONAL
            )
            
            # Create metadata object
            metadata = ChunkMetadata(
                section_id=raw_metadata.get("section_id", "unknown"),
                section_type=section_type,
                section_title=raw_metadata.get("section_title", ""),
                full_reference=raw_metadata.get("full_reference", ""),
                cfr_citation=raw_metadata.get("cfr_citation"),
                parent_section=raw_metadata.get("parent_section"),
                hierarchy_level=raw_metadata.get("hierarchy_level", 1),
                chunk_index=raw_metadata.get("chunk_index", 0),
                total_chunks=raw_metadata.get("total_chunks", 1),
                chunk_size=raw_metadata.get("chunk_size", 0),
                word_count=raw_metadata.get("word_count", 0),
                contains_definitions=raw_metadata.get("contains_definitions", False),
                contains_penalties=raw_metadata.get("contains_penalties", False),
                contains_requirements=raw_metadata.get("contains_requirements", False),
                references=raw_metadata.get("references", []),
                key_terms=raw_metadata.get("key_terms", []),
                content_type=content_type,
                compliance_level=compliance_level,
            )
            
            # Create DocumentChunk
            chunk = DocumentChunk(
                chunk_id=raw_chunk.get("chunk_id", 0),
                content=raw_chunk.get("content", ""),
                metadata=metadata,
                embedding=None,  # Will be generated later if requested
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )
            
            return chunk
            
        except Exception as e:
            raise ValueError(f"Failed to convert raw chunk: {e}")
            
    async def _add_embeddings_to_chunks(
        self,
        chunks: List[DocumentChunk],
        batch_size: int = 100
    ) -> None:
        """Generate embeddings for chunks."""
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        
        # Extract content texts
        texts = [chunk.content for chunk in chunks]
        
        # Generate embeddings in batches
        try:
            embeddings = await self.embedding_service.embed_texts_batch(
                texts, batch_size
            )
            
            # Assign embeddings to chunks
            embedded_count = 0
            for i, embedding in enumerate(embeddings):
                if embedding is not None:
                    chunks[i].embedding = embedding
                    embedded_count += 1
                else:
                    logger.warning(f"No embedding generated for chunk {i}")
                    
            logger.info(f"Generated embeddings for {embedded_count}/{len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {e}")
            raise
            
    async def get_ingestion_status(self) -> dict:
        """Get status of current ingestion."""
        total_chunks = await self.repository.count_chunks()
        embedded_chunks = await self.repository.count_chunks_with_embeddings()
        sections_summary = await self.repository.get_sections_summary()
        
        return {
            "total_chunks": total_chunks,
            "embedded_chunks": embedded_chunks,
            "embedding_coverage": embedded_chunks / total_chunks if total_chunks > 0 else 0,
            "sections_summary": sections_summary,
        }
        
    async def recompute_embeddings(
        self,
        batch_size: int = 100,
        section_filter: Optional[str] = None
    ) -> Tuple[int, int]:
        """
        Recompute embeddings for existing chunks.
        
        Args:
            batch_size: Batch size for processing
            section_filter: Optional section ID to filter chunks
            
        Returns:
            Tuple of (success_count, error_count)
        """
        logger.info("Starting embedding recomputation...")
        
        # Get chunks to recompute
        if section_filter:
            chunks_data = await self.repository.get_chunks_by_section(section_filter)
        else:
            # For now, we'll implement a simple approach
            # In a real implementation, you'd want pagination for large datasets
            logger.warning("Recomputing all embeddings - this may take a while")
            # This would need a method to get all chunks in batches
            raise NotImplementedError("Full recomputation not implemented yet")
            
        if not chunks_data:
            logger.info("No chunks found for recomputation")
            return 0, 0
            
        # Convert to DocumentChunk objects and generate embeddings
        chunks = []
        for chunk_data in chunks_data:
            # Convert database model back to schema
            # This is a simplified conversion - you might need more complete mapping
            metadata = ChunkMetadata(
                section_id=chunk_data.section_id,
                section_type=SectionType(chunk_data.section_type),
                section_title=chunk_data.section_title,
                full_reference=chunk_data.full_reference,
                cfr_citation=chunk_data.cfr_citation,
                parent_section=chunk_data.parent_section,
                hierarchy_level=chunk_data.hierarchy_level,
                chunk_index=chunk_data.chunk_index,
                total_chunks=chunk_data.total_chunks,
                chunk_size=chunk_data.chunk_size,
                word_count=chunk_data.word_count,
                contains_definitions=chunk_data.contains_definitions,
                contains_penalties=chunk_data.contains_penalties,
                contains_requirements=chunk_data.contains_requirements,
                references=chunk_data.references or [],
                key_terms=chunk_data.key_terms or [],
                content_type=ContentType(chunk_data.content_type),
                compliance_level=ComplianceLevel(chunk_data.compliance_level),
            )
            
            chunk = DocumentChunk(
                chunk_id=chunk_data.chunk_id,
                content=chunk_data.content,
                metadata=metadata,
                embedding=None,
                created_at=chunk_data.created_at,
                updated_at=datetime.utcnow(),
            )
            chunks.append(chunk)
            
        # Generate new embeddings
        await self._add_embeddings_to_chunks(chunks, batch_size)
        
        # Update in database
        success_count, error_count = await self.repository.bulk_insert_chunks(
            chunks, batch_size
        )
        
        logger.info(f"Embedding recomputation completed: {success_count} success, {error_count} errors")
        return success_count, error_count
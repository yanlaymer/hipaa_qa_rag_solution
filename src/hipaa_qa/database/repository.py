"""Repository pattern for database operations."""

import json
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
from loguru import logger
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from ..schemas import ChunkMetadata, ContentType, DocumentChunk
from .connection import DatabaseManager
from .models import DocumentChunkTable


class ChunkRepository:
    """Repository for document chunk operations."""
    
    def __init__(self, db_manager: DatabaseManager) -> None:
        """Initialize repository with database manager."""
        self.db_manager = db_manager
        
    async def insert_chunk(
        self, 
        chunk: DocumentChunk,
        session: Optional[AsyncSession] = None
    ) -> int:
        """Insert a single document chunk."""
        chunk_data = DocumentChunkTable(
            chunk_id=chunk.chunk_id,
            content=chunk.content,
            section_id=chunk.metadata.section_id,
            section_type=chunk.metadata.section_type.value,
            section_title=chunk.metadata.section_title,
            full_reference=chunk.metadata.full_reference,
            cfr_citation=chunk.metadata.cfr_citation,
            parent_section=chunk.metadata.parent_section,
            hierarchy_level=chunk.metadata.hierarchy_level,
            chunk_index=chunk.metadata.chunk_index,
            total_chunks=chunk.metadata.total_chunks,
            chunk_size=chunk.metadata.chunk_size,
            word_count=chunk.metadata.word_count,
            contains_definitions=chunk.metadata.contains_definitions,
            contains_penalties=chunk.metadata.contains_penalties,
            contains_requirements=chunk.metadata.contains_requirements,
            content_type=chunk.metadata.content_type.value,
            compliance_level=chunk.metadata.compliance_level.value,
            references=chunk.metadata.references,
            key_terms=chunk.metadata.key_terms,
            embedding=chunk.embedding,
            created_at=chunk.created_at,
            updated_at=chunk.updated_at,
        )
        
        if session:
            session.add(chunk_data)
            await session.flush()
            return chunk_data.id
        else:
            async with self.db_manager.get_session() as session:
                session.add(chunk_data)
                await session.commit()
                return chunk_data.id
                
    async def bulk_insert_chunks(
        self, 
        chunks: List[DocumentChunk],
        batch_size: int = 100
    ) -> Tuple[int, int]:
        """
        Bulk insert document chunks with embeddings.
        Returns (success_count, error_count).
        """
        success_count = 0
        error_count = 0
        
        logger.info(f"Starting bulk insert of {len(chunks)} chunks...")
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            try:
                async with self.db_manager.get_connection() as conn:
                    # Prepare data for insertion
                    chunk_data = []
                    for chunk in batch:
                        chunk_data.append((
                            chunk.chunk_id,
                            chunk.content,
                            chunk.metadata.section_id,
                            chunk.metadata.section_type.value,
                            chunk.metadata.section_title,
                            chunk.metadata.full_reference,
                            chunk.metadata.cfr_citation,
                            chunk.metadata.parent_section,
                            chunk.metadata.hierarchy_level,
                            chunk.metadata.chunk_index,
                            chunk.metadata.total_chunks,
                            chunk.metadata.chunk_size,
                            chunk.metadata.word_count,
                            chunk.metadata.contains_definitions,
                            chunk.metadata.contains_penalties,
                            chunk.metadata.contains_requirements,
                            chunk.metadata.content_type.value,
                            chunk.metadata.compliance_level.value,
                            json.dumps(chunk.metadata.references),
                            json.dumps(chunk.metadata.key_terms),
                            f"[{','.join(map(str, chunk.embedding))}]" if chunk.embedding else None,  # Format for pgvector
                            chunk.created_at,
                            chunk.updated_at,
                        ))
                    
                    # Execute bulk insert
                    await conn.executemany(
                        """
                        INSERT INTO document_chunks (
                            chunk_id, content, section_id, section_type, section_title,
                            full_reference, cfr_citation, parent_section, hierarchy_level,
                            chunk_index, total_chunks, chunk_size, word_count,
                            contains_definitions, contains_penalties, contains_requirements,
                            content_type, compliance_level, references_json, key_terms_json,
                            embedding, created_at, updated_at
                        ) VALUES (
                            $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                            $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
                            $21, $22, $23
                        )
                        ON CONFLICT (chunk_id) DO UPDATE SET
                            content = EXCLUDED.content,
                            embedding = EXCLUDED.embedding,
                            updated_at = EXCLUDED.updated_at
                        """,
                        chunk_data
                    )
                    
                success_count += len(batch)
                logger.info(f"Inserted batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
                
            except Exception as e:
                error_count += len(batch)
                logger.error(f"Failed to insert batch {i//batch_size + 1}: {e}")
                
        logger.info(f"Bulk insert completed: {success_count} success, {error_count} errors")
        return success_count, error_count
        
    async def similarity_search(
        self,
        query_embedding: List[float],
        limit: int = 5,
        similarity_threshold: float = 0.7,
        content_types: Optional[List[ContentType]] = None,
        sections: Optional[List[str]] = None,
    ) -> List[Tuple[DocumentChunkTable, float]]:
        """
        Perform similarity search using pgvector.
        Returns list of (chunk, similarity_score) tuples.
        """
        async with self.db_manager.get_connection() as conn:
            # Build query conditions
            where_conditions = ["embedding IS NOT NULL"]
            params = [query_embedding, limit]
            param_count = 2
            
            if content_types:
                param_count += 1
                where_conditions.append(f"content_type = ANY(${param_count})")
                params.append([ct.value for ct in content_types])
                
            if sections:
                param_count += 1
                where_conditions.append(f"section_id = ANY(${param_count})")
                params.append(sections)
            
            where_clause = " AND ".join(where_conditions)
            
            # Construct similarity search query
            query = f"""
                SELECT 
                    id, chunk_id, content, section_id, section_type, section_title,
                    full_reference, cfr_citation, parent_section, hierarchy_level,
                    chunk_index, total_chunks, chunk_size, word_count,
                    contains_definitions, contains_penalties, contains_requirements,
                    content_type, compliance_level, references_json, key_terms_json,
                    created_at, updated_at, metadata_json,
                    1 - (embedding <=> $1) as similarity_score
                FROM document_chunks
                WHERE {where_clause}
                ORDER BY embedding <=> $1
                LIMIT $2
            """
            
            # Format query vector for pgvector
            query_vector_str = f"[{','.join(map(str, query_embedding))}]"
            formatted_params = [query_vector_str] + params[1:]
            rows = await conn.fetch(query, *formatted_params)
            
            # Filter by similarity threshold and convert to objects
            results = []
            for row in rows:
                similarity_score = row['similarity_score']
                if similarity_score >= similarity_threshold:
                    # Convert row to DocumentChunkTable object
                    chunk = DocumentChunkTable(
                        id=row['id'],
                        chunk_id=row['chunk_id'],
                        content=row['content'],
                        section_id=row['section_id'],
                        section_type=row['section_type'],
                        section_title=row['section_title'],
                        full_reference=row['full_reference'],
                        cfr_citation=row['cfr_citation'],
                        parent_section=row['parent_section'],
                        hierarchy_level=row['hierarchy_level'],
                        chunk_index=row['chunk_index'],
                        total_chunks=row['total_chunks'],
                        chunk_size=row['chunk_size'],
                        word_count=row['word_count'],
                        contains_definitions=row['contains_definitions'],
                        contains_penalties=row['contains_penalties'],
                        contains_requirements=row['contains_requirements'],
                        content_type=row['content_type'],
                        compliance_level=row['compliance_level'],
                        references=row['references_json'] or [],
                        key_terms=row['key_terms_json'] or [],
                        created_at=row['created_at'],
                        updated_at=row['updated_at'],
                        metadata_json=row['metadata_json'],
                    )
                    results.append((chunk, similarity_score))
                    
            logger.info(
                f"Similarity search returned {len(results)} chunks "
                f"above threshold {similarity_threshold}"
            )
            return results
            
    async def get_chunk_by_id(self, chunk_id: int) -> Optional[DocumentChunkTable]:
        """Get a chunk by its chunk_id."""
        async with self.db_manager.get_session() as session:
            stmt = select(DocumentChunkTable).where(
                DocumentChunkTable.chunk_id == chunk_id
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()
            
    async def get_chunks_by_section(
        self, 
        section_id: str
    ) -> List[DocumentChunkTable]:
        """Get all chunks for a specific section."""
        async with self.db_manager.get_session() as session:
            stmt = select(DocumentChunkTable).where(
                DocumentChunkTable.section_id == section_id
            ).order_by(DocumentChunkTable.chunk_index)
            result = await session.execute(stmt)
            return list(result.scalars().all())
            
    async def count_chunks(self) -> int:
        """Get total number of chunks in database."""
        async with self.db_manager.get_session() as session:
            stmt = select(text("COUNT(*)")).select_from(DocumentChunkTable)
            result = await session.execute(stmt)
            return result.scalar() or 0
            
    async def count_chunks_with_embeddings(self) -> int:
        """Get number of chunks that have embeddings."""
        async with self.db_manager.get_session() as session:
            stmt = select(text("COUNT(*)")).select_from(DocumentChunkTable).where(
                DocumentChunkTable.embedding.is_not(None)
            )
            result = await session.execute(stmt)
            return result.scalar() or 0
            
    async def delete_all_chunks(self) -> int:
        """Delete all chunks from database. Returns number of deleted rows."""
        async with self.db_manager.get_session() as session:
            stmt = delete(DocumentChunkTable)
            result = await session.execute(stmt)
            await session.commit()
            deleted_count = result.rowcount or 0
            logger.info(f"Deleted {deleted_count} chunks from database")
            return deleted_count
            
    async def get_sections_summary(self) -> Dict[str, Any]:
        """Get summary statistics about sections in the database."""
        async with self.db_manager.get_connection() as conn:
            query = """
                SELECT 
                    section_type,
                    COUNT(*) as chunk_count,
                    COUNT(CASE WHEN embedding IS NOT NULL THEN 1 END) as embedded_count,
                    COUNT(CASE WHEN contains_definitions THEN 1 END) as definition_count,
                    COUNT(CASE WHEN contains_requirements THEN 1 END) as requirement_count,
                    COUNT(CASE WHEN contains_penalties THEN 1 END) as penalty_count
                FROM document_chunks
                GROUP BY section_type
                ORDER BY section_type
            """
            
            rows = await conn.fetch(query)
            
            summary = {
                "by_section_type": {},
                "total_chunks": 0,
                "total_embedded": 0,
            }
            
            for row in rows:
                section_type = row['section_type']
                summary["by_section_type"][section_type] = {
                    "chunk_count": row['chunk_count'],
                    "embedded_count": row['embedded_count'],
                    "definition_count": row['definition_count'],
                    "requirement_count": row['requirement_count'],
                    "penalty_count": row['penalty_count'],
                }
                summary["total_chunks"] += row['chunk_count']
                summary["total_embedded"] += row['embedded_count']
                
            return summary
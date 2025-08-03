#!/usr/bin/env python3
"""
Enhanced Data Ingestion Script

This script ingests the enhanced HIPAA chunks with improved metadata
into the PostgreSQL database with pgvector for semantic search.
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hipaa_qa.config import Settings
from hipaa_qa.database import DatabaseManager, ChunkRepository, wait_for_database
from hipaa_qa.services import EmbeddingService
from hipaa_qa.schemas import DocumentChunk, ChunkMetadata, SectionType, ContentType, ComplianceLevel

# Enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_ingestion.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


async def ingest_enhanced_chunks():
    """Ingest enhanced HIPAA chunks into the database."""
    logger.info("🚀 Starting enhanced data ingestion...")
    
    # Load settings
    settings = Settings()
    
    # Wait for database
    logger.info("⏳ Waiting for database connection...")
    db_available = await wait_for_database(max_retries=30, retry_interval=2.0, settings=settings)
    if not db_available:
        raise RuntimeError("Database connection failed")
    
    # Initialize services
    db_manager = DatabaseManager(settings)
    embedding_service = EmbeddingService(settings)
    
    try:
        # Load enhanced chunks
        chunks_file = Path("data/clean/enhanced_hipaa_chunks.json")
        if not chunks_file.exists():
            raise FileNotFoundError(f"Enhanced chunks file not found: {chunks_file}")
        
        logger.info(f"📖 Loading enhanced chunks from {chunks_file}")
        with open(chunks_file, 'r', encoding='utf-8') as f:
            enhanced_data = json.load(f)
        
        logger.info(f"📋 Loaded {len(enhanced_data)} enhanced chunks")
        
        # Clear existing data
        repository = ChunkRepository(db_manager)
        logger.info("🧹 Clearing existing chunks...")
        deleted_count = await repository.delete_all_chunks()
        logger.info(f"🗑️ Deleted {deleted_count} existing chunks")
        
        # Process chunks in batches
        batch_size = 50
        total_batches = (len(enhanced_data) + batch_size - 1) // batch_size
        
        all_chunks = []
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(enhanced_data))
            batch_data = enhanced_data[start_idx:end_idx]
            
            logger.info(f"🔄 Processing batch {batch_idx + 1}/{total_batches} (chunks {start_idx}-{end_idx-1})")
            
            # Generate embeddings for batch
            batch_texts = [chunk['content'] for chunk in batch_data]
            batch_embeddings = await embedding_service.embed_texts_batch(batch_texts)
            
            # Create DocumentChunk objects
            batch_chunks = []
            for i, (chunk_data, embedding) in enumerate(zip(batch_data, batch_embeddings)):
                metadata_dict = chunk_data['metadata']
                
                # Map section type
                section_type_map = {
                    'part': SectionType.PART,
                    'subpart': SectionType.SUBPART,
                    'section': SectionType.SECTION,
                    'subsection': SectionType.SUBSECTION,
                    'paragraph': SectionType.PARAGRAPH
                }
                section_type = section_type_map.get(metadata_dict.get('section_type', 'section'), SectionType.SECTION)
                
                # Enhanced content categorization
                content_categories = metadata_dict.get('content_categories', [])
                if 'definitions' in content_categories:
                    content_type = ContentType.DEFINITION
                elif 'requirements' in content_categories:
                    content_type = ContentType.REQUIREMENT
                elif 'penalties' in content_categories:
                    content_type = ContentType.PENALTY
                else:
                    content_type = ContentType.GENERAL
                
                # Build references and key terms from metadata
                references = []
                if metadata_dict.get('cfr_citation'):
                    references.append(metadata_dict['cfr_citation'])
                if metadata_dict.get('section_id'):
                    references.append(f"§ {metadata_dict['section_id']}")
                
                key_terms = list(metadata_dict.get('key_concepts', []))
                
                # Create proper ChunkMetadata
                chunk_metadata = ChunkMetadata(
                    section_id=metadata_dict['section_id'],
                    section_type=section_type,
                    section_title=metadata_dict['section_title'],
                    full_reference=metadata_dict.get('full_reference', f"45 CFR § {metadata_dict['section_id']}"),
                    cfr_citation=metadata_dict.get('cfr_citation'),
                    parent_section=metadata_dict.get('parent_section'),
                    hierarchy_level=metadata_dict.get('hierarchy_level', 3),
                    chunk_index=metadata_dict.get('chunk_index', 0),
                    total_chunks=metadata_dict.get('total_chunks', 1),
                    chunk_size=len(chunk_data['content']),
                    word_count=metadata_dict.get('word_count', len(chunk_data['content'].split())),
                    contains_definitions='definitions' in content_categories,
                    contains_penalties='penalties' in content_categories,
                    contains_requirements='requirements' in content_categories,
                    references=references,
                    key_terms=key_terms,
                    content_type=content_type,
                    compliance_level=ComplianceLevel.INFORMATIONAL
                )
                
                chunk = DocumentChunk(
                    chunk_id=chunk_data['chunk_id'],
                    content=chunk_data['content'],
                    metadata=chunk_metadata,
                    embedding=embedding
                )
                batch_chunks.append(chunk)
            
            all_chunks.extend(batch_chunks)
            
            # Insert batch into database
            try:
                await repository.bulk_insert_chunks(batch_chunks)
                logger.info(f"✅ Successfully inserted batch {batch_idx + 1}")
            except Exception as e:
                logger.error(f"❌ Failed to insert batch {batch_idx + 1}: {e}")
                raise
        
        # Verify insertion
        total_inserted = await repository.count_chunks()
        logger.info(f"📊 Verification: {total_inserted} chunks in database")
        
        if total_inserted != len(enhanced_data):
            logger.warning(f"⚠️ Mismatch: Expected {len(enhanced_data)}, got {total_inserted}")
        else:
            logger.info("✅ All enhanced chunks successfully ingested!")
        
        # Test retrieval with a sample query
        logger.info("🔍 Testing enhanced retrieval...")
        test_embedding = await embedding_service.embed_text("What is protected health information?")
        test_results = await repository.similarity_search(
            query_embedding=test_embedding,
            limit=3,
            similarity_threshold=0.1
        )
        
        logger.info(f"🎯 Test retrieval returned {len(test_results)} results")
        for i, result in enumerate(test_results):
            chunk_id, content, section_id, cfr_citation, references_json, key_terms_json, similarity = result
            logger.info(f"  {i+1}. {cfr_citation or section_id} (similarity: {similarity:.3f})")
        
    except Exception as e:
        logger.error(f"❌ Enhanced ingestion failed: {e}")
        raise
    finally:
        await db_manager.close()
    
    logger.info("🎉 Enhanced data ingestion completed successfully!")


async def main():
    """Main function."""
    try:
        await ingest_enhanced_chunks()
    except KeyboardInterrupt:
        logger.info("⏹️ Ingestion interrupted by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
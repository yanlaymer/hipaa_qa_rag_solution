"""Question-answering API endpoints."""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from ...schemas import AnswerResponse, ContentType, QuestionRequest, SearchRequest, SearchResponse
from ..dependencies import get_app_state
from ..state import AppState

router = APIRouter()


@router.post("/ask", response_model=AnswerResponse, summary="Ask a question about HIPAA")
async def ask_question(
    request: QuestionRequest,
    app_state: AppState = Depends(get_app_state)
):
    """
    Ask a question about HIPAA regulations and get an answer with citations.
    
    This endpoint uses a RAG (Retrieval-Augmented Generation) pipeline to:
    1. Find relevant regulation text chunks using semantic similarity
    2. Generate an answer using OpenAI's GPT model with proper citations
    3. Return the answer along with source references
    
    Args:
        request: Question request with the user's question and optional parameters
        
    Returns:
        AnswerResponse: Generated answer with citations and source references
        
    Raises:
        HTTPException: 400 for invalid requests, 500 for processing errors
    """
    try:
        logger.info(f"Processing question: {request.question[:100]}...")
        
        # Validate request
        if not request.question.strip():
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty"
            )
            
        # Process the question using QA service
        response = await app_state.qa_service.answer_question(
            question=request.question,
            max_chunks=request.max_chunks or app_state.settings.max_chunks_retrieved,
            similarity_threshold=request.similarity_threshold or app_state.settings.similarity_threshold,
        )
        
        logger.info(
            f"Question processed successfully: "
            f"{response.chunks_retrieved} chunks, "
            f"{response.processing_time_ms}ms, "
            f"confidence: {response.confidence_score}"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing question: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process question: {str(e)}"
        )


@router.post("/search", response_model=SearchResponse, summary="Search regulation chunks")
async def search_chunks(
    request: SearchRequest,
    app_state: AppState = Depends(get_app_state)
):
    """
    Search for regulation chunks using semantic similarity.
    
    This endpoint performs semantic search without generating an answer,
    useful for exploring the knowledge base or debugging retrieval.
    
    Args:
        request: Search request with query and filters
        
    Returns:
        SearchResponse: List of matching chunks with similarity scores
        
    Raises:
        HTTPException: 400 for invalid requests, 500 for processing errors
    """
    try:
        import time
        start_time = time.time()
        
        logger.info(f"Searching for: {request.query[:100]}...")
        
        # Generate query embedding
        query_embedding = await app_state.embedding_service.embed_text(request.query)
        
        # Perform similarity search
        chunk_results = await app_state.qa_service.repository.similarity_search(
            query_embedding=query_embedding,
            limit=request.max_results,
            similarity_threshold=request.similarity_threshold,
            content_types=request.filter_content_types,
            sections=request.filter_sections,
        )
        
        # Build search results
        from ...schemas import SearchResult, ChunkMetadata, SectionType, ContentType, ComplianceLevel
        
        results = []
        for chunk, similarity_score in chunk_results:
            # Convert database model to schema
            metadata = ChunkMetadata(
                section_id=chunk.section_id,
                section_type=SectionType(chunk.section_type),
                section_title=chunk.section_title,
                full_reference=chunk.full_reference,
                cfr_citation=chunk.cfr_citation,
                parent_section=chunk.parent_section,
                hierarchy_level=chunk.hierarchy_level,
                chunk_index=chunk.chunk_index,
                total_chunks=chunk.total_chunks,
                chunk_size=chunk.chunk_size,
                word_count=chunk.word_count,
                contains_definitions=chunk.contains_definitions,
                contains_penalties=chunk.contains_penalties,
                contains_requirements=chunk.contains_requirements,
                references=chunk.references or [],
                key_terms=chunk.key_terms or [],
                content_type=ContentType(chunk.content_type),
                compliance_level=ComplianceLevel(chunk.compliance_level),
            )
            
            result = SearchResult(
                chunk_id=chunk.chunk_id,
                content=chunk.content,
                similarity_score=similarity_score,
                metadata=metadata,
            )
            results.append(result)
            
        # Calculate search time
        search_time_ms = int((time.time() - start_time) * 1000)
        
        response = SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            search_time_ms=search_time_ms,
            similarity_threshold_used=request.similarity_threshold,
        )
        
        logger.info(f"Search completed: {len(results)} results in {search_time_ms}ms")
        return response
        
    except Exception as e:
        logger.error(f"Error performing search: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to perform search: {str(e)}"
        )


@router.get("/models", summary="Get model information")
async def get_model_info(app_state: AppState = Depends(get_app_state)):
    """
    Get information about the models being used for embeddings and chat.
    
    Returns:
        dict: Model information and configuration
    """
    try:
        embedding_info = app_state.embedding_service.get_model_info()
        
        return {
            "embedding_model": embedding_info,
            "chat_model": {
                "model": app_state.settings.openai_chat_model,
                "provider": "OpenAI",
                "max_tokens": 4096,  # GPT-4 context length
            },
            "configuration": {
                "embedding_dimension": app_state.settings.embedding_dimension,
                "similarity_threshold": app_state.settings.similarity_threshold,
                "max_chunks_retrieved": app_state.settings.max_chunks_retrieved,
                "batch_size": app_state.settings.batch_size,
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get model information: {str(e)}"
        )


@router.get("/content-types", summary="Get available content types")
async def get_content_types():
    """
    Get list of available content types for filtering.
    
    Returns:
        dict: Available content types and their descriptions
    """
    return {
        "content_types": [
            {
                "value": ContentType.DEFINITION.value,
                "description": "Regulatory definitions and terms"
            },
            {
                "value": ContentType.REQUIREMENT.value,
                "description": "Compliance requirements and obligations"
            },
            {
                "value": ContentType.GENERAL.value,
                "description": "General regulatory text"
            },
            {
                "value": ContentType.PENALTY.value,
                "description": "Penalties and enforcement information"
            },
            {
                "value": ContentType.PROCEDURE.value,
                "description": "Procedures and processes"
            },
        ]
    }


@router.get("/sections", summary="Get available sections")
async def get_sections(app_state: AppState = Depends(get_app_state)):
    """
    Get summary of available regulation sections.
    
    Returns:
        dict: Available sections and statistics
    """
    try:
        sections_summary = await app_state.ingestion_service.repository.get_sections_summary()
        return sections_summary
        
    except Exception as e:
        logger.error(f"Error getting sections: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get sections: {str(e)}"
        )
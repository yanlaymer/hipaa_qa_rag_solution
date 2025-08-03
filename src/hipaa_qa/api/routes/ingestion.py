"""Data ingestion API endpoints."""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from loguru import logger

from ...schemas import BulkIngestionRequest, IngestionResponse
from ..dependencies import get_app_state
from ..state import AppState

router = APIRouter()


@router.post("/ingest", response_model=IngestionResponse, summary="Ingest document chunks")
async def ingest_chunks(
    request: BulkIngestionRequest,
    background_tasks: BackgroundTasks,
    app_state: AppState = Depends(get_app_state)
):
    """
    Ingest document chunks from JSON file into the database.
    
    This endpoint processes the HIPAA regulation chunks, generates embeddings,
    and stores them in the vector database for semantic search.
    
    Args:
        request: Ingestion request with file path and options
        background_tasks: FastAPI background tasks for async processing
        
    Returns:
        IngestionResponse: Processing status and statistics
        
    Raises:
        HTTPException: 400 for invalid requests, 500 for processing errors
    """
    try:
        import time
        start_time = time.time()
        
        logger.info(f"Starting ingestion from {request.source_file}")
        
        # Validate source file path
        if not request.source_file:
            raise HTTPException(
                status_code=400,
                detail="Source file path is required"
            )
            
        # Perform ingestion
        success_count, error_count, errors = await app_state.ingestion_service.ingest_from_json(
            json_file_path=request.source_file,
            batch_size=request.batch_size,
            overwrite_existing=request.overwrite_existing,
            generate_embeddings=True,  # Always generate embeddings for new ingestion
        )
        
        # Calculate processing time
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        # Determine status
        if error_count == 0:
            status = "completed"
        elif success_count > error_count:
            status = "completed_with_errors"
        else:
            status = "failed"
            
        response = IngestionResponse(
            status=status,
            chunks_processed=success_count + error_count,
            chunks_inserted=success_count,
            chunks_failed=error_count,
            processing_time_ms=processing_time_ms,
            errors=errors[:10],  # Limit to first 10 errors
        )
        
        logger.info(
            f"Ingestion completed: {success_count} success, {error_count} errors, "
            f"{processing_time_ms}ms"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during ingestion: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed: {str(e)}"
        )


@router.post("/recompute-embeddings", summary="Recompute embeddings for existing chunks")
async def recompute_embeddings(
    section_filter: str = None,
    batch_size: int = 100,
    app_state: AppState = Depends(get_app_state)
):
    """
    Recompute embeddings for existing chunks.
    
    Useful when switching embedding models or fixing embedding issues.
    
    Args:
        section_filter: Optional section ID to filter chunks
        batch_size: Batch size for processing
        
    Returns:
        dict: Recomputation results
        
    Raises:
        HTTPException: 500 for processing errors
    """
    try:
        import time
        start_time = time.time()
        
        logger.info(f"Starting embedding recomputation (section: {section_filter})")
        
        success_count, error_count = await app_state.ingestion_service.recompute_embeddings(
            batch_size=batch_size,
            section_filter=section_filter,
        )
        
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        logger.info(
            f"Embedding recomputation completed: {success_count} success, "
            f"{error_count} errors, {processing_time_ms}ms"
        )
        
        return {
            "status": "completed" if error_count == 0 else "completed_with_errors",
            "chunks_recomputed": success_count,
            "chunks_failed": error_count,
            "processing_time_ms": processing_time_ms,
            "section_filter": section_filter,
        }
        
    except Exception as e:
        logger.error(f"Error during embedding recomputation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Embedding recomputation failed: {str(e)}"
        )


@router.get("/status", summary="Get ingestion status")
async def get_ingestion_status(app_state: AppState = Depends(get_app_state)):
    """
    Get current ingestion status and statistics.
    
    Returns:
        dict: Ingestion status information
    """
    try:
        status = await app_state.ingestion_service.get_ingestion_status()
        return status
        
    except Exception as e:
        logger.error(f"Error getting ingestion status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get ingestion status: {str(e)}"
        )


@router.delete("/clear", summary="Clear all ingested data")
async def clear_ingested_data(
    confirm: bool = False,
    app_state: AppState = Depends(get_app_state)
):
    """
    Clear all ingested document chunks from the database.
    
    ⚠️ WARNING: This operation is destructive and cannot be undone!
    
    Args:
        confirm: Must be True to proceed with deletion
        
    Returns:
        dict: Deletion results
        
    Raises:
        HTTPException: 400 if not confirmed, 500 for processing errors
    """
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Must set confirm=true to clear all data. This operation cannot be undone."
        )
        
    try:
        logger.warning("Clearing all ingested data...")
        
        deleted_count = await app_state.ingestion_service.repository.delete_all_chunks()
        
        logger.warning(f"Cleared {deleted_count} chunks from database")
        
        return {
            "status": "completed",
            "chunks_deleted": deleted_count,
            "message": "All ingested data has been cleared"
        }
        
    except Exception as e:
        logger.error(f"Error clearing data: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear data: {str(e)}"
        )
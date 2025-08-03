"""Health check endpoints."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from ...schemas import HealthResponse
from ..dependencies import get_app_state
from ..state import AppState

router = APIRouter()


@router.get("/", response_model=HealthResponse, summary="Basic health check")
async def health_check(app_state: AppState = Depends(get_app_state)):
    """
    Basic health check endpoint.
    
    Returns:
        HealthResponse: Current service health status
    """
    try:
        # Check database connection
        db_connected = await app_state.db_manager.check_connection()
        
        # Check OpenAI API access
        openai_accessible = await app_state.embedding_service.validate_api_access()
        
        # Get chunk count
        chunks_indexed = await app_state.db_manager.get_chunks_count()
        
        # Determine overall status
        if db_connected and openai_accessible and chunks_indexed > 0:
            status = "healthy"
        elif db_connected and chunks_indexed > 0:
            status = "degraded"  # API issues but core functionality works
        else:
            status = "unhealthy"
            
        return HealthResponse(
            status=status,
            timestamp=datetime.utcnow(),
            version="0.1.0",
            database_connected=db_connected,
            openai_accessible=openai_accessible,
            chunks_indexed=chunks_indexed,
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.utcnow(),
            version="0.1.0",
            database_connected=False,
            openai_accessible=False,
            chunks_indexed=0,
        )


@router.get("/detailed", summary="Detailed health information")
async def detailed_health(app_state: AppState = Depends(get_app_state)):
    """
    Detailed health information including database and service statistics.
    
    Returns:
        dict: Detailed health and status information
    """
    try:
        # Basic health check
        basic_health = await health_check(app_state)
        
        # Additional detailed information
        ingestion_status = await app_state.ingestion_service.get_ingestion_status()
        sections_summary = await app_state.ingestion_service.repository.get_sections_summary()
        
        # Embedding model info
        embedding_info = app_state.embedding_service.get_model_info()
        
        return {
            "basic_health": basic_health.dict(),
            "ingestion_status": ingestion_status,
            "sections_summary": sections_summary,
            "embedding_model": embedding_info,
            "settings": {
                "embedding_dimension": app_state.settings.embedding_dimension,
                "similarity_threshold": app_state.settings.similarity_threshold,
                "max_chunks_retrieved": app_state.settings.max_chunks_retrieved,
                "chat_model": app_state.settings.openai_chat_model,
            }
        }
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/ready", summary="Readiness probe")
async def readiness_check(app_state: AppState = Depends(get_app_state)):
    """
    Kubernetes-style readiness probe.
    
    Returns:
        dict: Simple ready/not ready status
        
    Raises:
        HTTPException: 503 if service is not ready
    """
    try:
        # Check if service is ready to handle requests
        db_connected = await app_state.db_manager.check_connection()
        chunks_count = await app_state.db_manager.get_chunks_count()
        
        if db_connected and chunks_count > 0:
            return {"status": "ready"}
        else:
            raise HTTPException(
                status_code=503,
                detail="Service not ready - database not connected or no data loaded"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Service not ready: {str(e)}"
        )


@router.get("/live", summary="Liveness probe")
async def liveness_check():
    """
    Kubernetes-style liveness probe.
    
    Returns:
        dict: Simple alive status
    """
    return {"status": "alive", "timestamp": datetime.utcnow()}
"""FastAPI application factory and setup."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from ..config import get_settings
from .middleware import LoggingMiddleware, RequestIDMiddleware
from .routes import health, ingestion, qa
from .state import get_app_state


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager."""
    logger.info("Starting HIPAA QA System...")
    
    try:
        # Get app state
        app_state = get_app_state()
        
        # Initialize database
        await app_state.db_manager.initialize_database()
        logger.info("Database initialized successfully")
        
        # Validate API access
        embedding_valid = await app_state.embedding_service.validate_api_access()
        chat_valid = await app_state.qa_service.validate_chat_api_access()
        
        if not embedding_valid:
            logger.warning("OpenAI Embedding API validation failed")
        if not chat_valid:
            logger.warning("OpenAI Chat API validation failed")
            
        # Log ingestion status
        status = await app_state.ingestion_service.get_ingestion_status()
        logger.info(f"Ingestion status: {status}")
        
        logger.info("HIPAA QA System startup complete")
        
        # Application is ready
        yield
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise
        
    finally:
        # Cleanup
        logger.info("Shutting down HIPAA QA System...")
        app_state = get_app_state()
        await app_state.db_manager.close()
        logger.info("Shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="HIPAA QA System",
        description="RAG-based Question Answering System for HIPAA Regulations",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    
    # Add middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(LoggingMiddleware)
    
    # Add exception handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc) if settings.api_debug else "An unexpected error occurred",
            }
        )
    
    # Include routers
    app.include_router(health.router, prefix="/health", tags=["Health"])
    app.include_router(qa.router, prefix="/qa", tags=["Question Answering"])
    app.include_router(ingestion.router, prefix="/ingestion", tags=["Data Ingestion"])
    
    # Root endpoint
    @app.get("/", summary="Root endpoint")
    async def root():
        """Root endpoint with basic API information."""
        return {
            "service": "HIPAA QA System",
            "version": "0.1.0",
            "description": "RAG-based Question Answering for HIPAA Regulations",
            "docs": "/docs",
            "health": "/health",
        }
    
    return app


# Global app instance
_app: FastAPI = None


def get_app() -> FastAPI:
    """Get the global FastAPI application instance."""
    global _app
    if _app is None:
        _app = create_app()
    return _app



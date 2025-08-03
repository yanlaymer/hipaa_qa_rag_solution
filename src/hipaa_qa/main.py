"""Main entry point for the HIPAA QA System backend."""

import sys
from pathlib import Path

import uvicorn
from loguru import logger

# Add src to path for imports
src_path = Path(__file__).parent.parent
sys.path.insert(0, str(src_path))

from hipaa_qa.api.main import get_app
from hipaa_qa.config import get_settings


def setup_logging() -> None:
    """Configure logging for the application."""
    settings = get_settings()
    
    # Remove default handler
    logger.remove()
    
    # Configure format based on settings
    if settings.log_format == "json":
        format_string = (
            '{{"time": "{time:YYYY-MM-DD HH:mm:ss.SSS}", '
            '"level": "{level}", '
            '"module": "{module}", '
            '"function": "{function}", '
            '"line": {line}, '
            '"message": "{message}"}}'
        )
    else:
        format_string = (
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{module}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
    
    # Add handler with configured format
    logger.add(
        sys.stdout,
        format=format_string,
        level=settings.log_level,
        colorize=settings.log_format == "text",
        serialize=settings.log_format == "json",
    )
    
    # Add file handler for errors
    logger.add(
        "logs/hipaa_qa_errors.log",
        format=format_string,
        level="ERROR",
        rotation="1 day",
        retention="7 days",
        serialize=settings.log_format == "json",
    )
    
    logger.info(f"Logging configured: level={settings.log_level}, format={settings.log_format}")


def main() -> None:
    """Main entry point."""
    # Setup logging first
    setup_logging()
    
    # Get settings
    settings = get_settings()
    
    logger.info("Starting HIPAA QA System backend...")
    logger.info(f"Configuration: host={settings.api_host}, port={settings.api_port}")
    
    # Get FastAPI app
    app = get_app()
    
    # Run with uvicorn
    uvicorn.run(
        app,
        host=settings.api_host,
        port=settings.api_port,
        workers=settings.api_workers,
        log_config=None,  # Use our custom logging
        access_log=False,  # We handle this in middleware
    )


if __name__ == "__main__":
    main()
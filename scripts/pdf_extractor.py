"""Async Production PDF Extractor for HIPAA QA System.

This module provides an async, production-ready PDF extraction service
using Mistral's OCR API with comprehensive error handling, logging,
and configuration management.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
import json
from datetime import datetime
import aiofiles
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from mistralai import Mistral
from mistralai import DocumentURLChunk, ImageURLChunk, TextChunk
import dotenv

# Load environment variables
dotenv.load_dotenv()


logger = logging.getLogger(__name__)


@dataclass
class ExtractionConfig:
    """Configuration for PDF extraction process."""
    input_pdf_path: Path
    output_text_path: Path
    mistral_api_key: str
    model: str = "mistral-ocr-latest"
    include_image_base64: bool = False
    signed_url_expiry_hours: int = 1
    max_retries: int = 3
    timeout_seconds: int = 300
    chunk_size: int = 1024 * 1024  # 1MB chunks for file upload

    @classmethod
    def from_env(cls, input_pdf_path: str, output_text_path: str) -> "ExtractionConfig":
        """Create configuration from environment variables."""
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY environment variable is required")
        
        return cls(
            input_pdf_path=Path(input_pdf_path),
            output_text_path=Path(output_text_path),
            mistral_api_key=api_key,
            model=os.getenv("MISTRAL_OCR_MODEL", "mistral-ocr-latest"),
            max_retries=int(os.getenv("MISTRAL_MAX_RETRIES", "3")),
            timeout_seconds=int(os.getenv("MISTRAL_TIMEOUT_SECONDS", "300"))
        )


@dataclass
class ExtractionProgress:
    """Progress tracking for PDF extraction."""
    stage: str
    progress_percent: float
    message: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ExtractionResult:
    """Result of PDF extraction process."""
    success: bool
    extracted_text: Optional[str]
    output_file_path: Optional[Path]
    error_message: Optional[str]
    processing_time_seconds: float
    pages_processed: int
    file_size_bytes: int
    metadata: Dict[str, Any]


class PDFExtractionError(Exception):
    """Custom exception for PDF extraction errors."""
    pass


class AsyncPDFExtractor:
    """Async production-ready PDF extractor using Mistral OCR."""
    
    def __init__(self, config: ExtractionConfig, progress_callback: Optional[Callable[[ExtractionProgress], None]] = None):
        self.config = config
        self.progress_callback = progress_callback
        self.client = Mistral(api_key=config.mistral_api_key)
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if not self.config.input_pdf_path.exists():
            raise PDFExtractionError(f"Input PDF file does not exist: {self.config.input_pdf_path}")
        
        if not self.config.input_pdf_path.suffix.lower() == '.pdf':
            raise PDFExtractionError(f"Input file is not a PDF: {self.config.input_pdf_path}")
        
        # Ensure output directory exists
        self.config.output_text_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✅ Configuration validated for PDF extraction: {self.config.input_pdf_path}")
    
    def _report_progress(self, stage: str, progress: float, message: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Report progress if callback is provided."""
        if self.progress_callback:
            progress_info = ExtractionProgress(
                stage=stage,
                progress_percent=progress,
                message=message,
                timestamp=datetime.now(),
                metadata=metadata or {}
            )
            self.progress_callback(progress_info)
        
        logger.info(f"📊 [{stage}] {progress:.1f}% - {message}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.RequestError, PDFExtractionError))
    )
    async def _upload_pdf_with_retry(self) -> Any:
        """Upload PDF file to Mistral with retry logic."""
        try:
            self._report_progress("upload", 10, "Reading PDF file")
            
            # Read file asynchronously
            async with aiofiles.open(self.config.input_pdf_path, 'rb') as f:
                file_content = await f.read()
            
            file_size_mb = len(file_content) / (1024 * 1024)
            self._report_progress("upload", 30, f"Uploading PDF file ({file_size_mb:.1f} MB)", 
                                {"file_size_bytes": len(file_content)})
            
            # Upload file
            uploaded_file = self.client.files.upload(
                file={
                    "file_name": self.config.input_pdf_path.stem,
                    "content": file_content,
                },
                purpose="ocr",
            )
            
            self._report_progress("upload", 60, f"File uploaded successfully. ID: {uploaded_file.id}")
            return uploaded_file
            
        except Exception as e:
            error_msg = f"Failed to upload PDF: {str(e)}"
            logger.error(error_msg)
            raise PDFExtractionError(error_msg) from e
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.RequestError, PDFExtractionError))
    )
    async def _process_ocr_with_retry(self, uploaded_file: Any) -> Any:
        """Process OCR with retry logic."""
        try:
            self._report_progress("processing", 70, "Getting signed URL for processing")
            
            # Get signed URL
            signed_url = self.client.files.get_signed_url(
                file_id=uploaded_file.id, 
                expiry=self.config.signed_url_expiry_hours
            )
            
            self._report_progress("processing", 80, "Starting OCR processing")
            
            # Process OCR
            pdf_response = self.client.ocr.process(
                document=DocumentURLChunk(document_url=signed_url.url),
                model=self.config.model,
                include_image_base64=self.config.include_image_base64
            )
            
            self._report_progress("processing", 90, f"OCR completed. Pages processed: {len(pdf_response.pages)}")
            return pdf_response
            
        except Exception as e:
            error_msg = f"Failed to process OCR: {str(e)}"
            logger.error(error_msg)
            raise PDFExtractionError(error_msg) from e
    
    async def _save_results(self, pdf_response: Any) -> int:
        """Save OCR results to file asynchronously."""
        try:
            self._report_progress("saving", 95, "Saving OCR results to file")
            
            async with aiofiles.open(self.config.output_text_path, "w", encoding="utf-8") as f:
                for i, page in enumerate(pdf_response.pages):
                    await f.write(page.markdown)
                    await f.write("\n\n---\n\n")  # Add separator between pages
                    
                    if i % 10 == 0:  # Progress update every 10 pages
                        progress = 95 + (i / len(pdf_response.pages)) * 5
                        self._report_progress("saving", progress, f"Saved page {i+1}/{len(pdf_response.pages)}")
            
            pages_count = len(pdf_response.pages)
            self._report_progress("saving", 100, f"Successfully saved {pages_count} pages to {self.config.output_text_path}")
            
            return pages_count
            
        except Exception as e:
            error_msg = f"Failed to save OCR results: {str(e)}"
            logger.error(error_msg)
            raise PDFExtractionError(error_msg) from e
    
    async def extract_async(self) -> ExtractionResult:
        """Extract text from PDF asynchronously with comprehensive error handling."""
        start_time = datetime.now()
        file_size = self.config.input_pdf_path.stat().st_size
        
        try:
            logger.info(f"🚀 Starting async PDF extraction: {self.config.input_pdf_path}")
            self._report_progress("init", 0, "Starting PDF extraction")
            
            # Upload PDF
            uploaded_file = await self._upload_pdf_with_retry()
            
            # Process OCR
            pdf_response = await self._process_ocr_with_retry(uploaded_file)
            
            # Save results
            pages_processed = await self._save_results(pdf_response)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Extract text from all pages
            extracted_text = ""
            for page in pdf_response.pages:
                extracted_text += page.markdown + "\n\n---\n\n"
            
            result = ExtractionResult(
                success=True,
                extracted_text=extracted_text.strip(),
                output_file_path=self.config.output_text_path,
                error_message=None,
                processing_time_seconds=processing_time,
                pages_processed=pages_processed,
                file_size_bytes=file_size,
                metadata={
                    "model_used": self.config.model,
                    "input_file": str(self.config.input_pdf_path),
                    "output_file": str(self.config.output_text_path),
                    "extraction_timestamp": start_time.isoformat()
                }
            )
            
            logger.info(f"✅ PDF extraction completed successfully in {processing_time:.2f}s")
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"PDF extraction failed: {str(e)}"
            logger.error(error_msg)
            
            result = ExtractionResult(
                success=False,
                extracted_text=None,
                output_file_path=None,
                error_message=error_msg,
                processing_time_seconds=processing_time,
                pages_processed=0,
                file_size_bytes=file_size,
                metadata={
                    "error_type": type(e).__name__,
                    "input_file": str(self.config.input_pdf_path),
                    "extraction_timestamp": start_time.isoformat()
                }
            )
            
            return result
    
    async def cleanup(self) -> None:
        """Cleanup any temporary resources."""
        # Add any cleanup logic here if needed
        logger.info("🧹 Cleanup completed")


# Legacy sync wrapper for backward compatibility
def extract_txt(pdf_path: str) -> str:
    """Legacy synchronous wrapper for backward compatibility.
    
    WARNING: This function is deprecated. Use AsyncPDFExtractor for new code.
    """
    logger.warning("extract_txt() is deprecated. Use AsyncPDFExtractor for new code.")
    
    config = ExtractionConfig.from_env(
        input_pdf_path=pdf_path,
        output_text_path="data/clean/hipaa_regulations.txt"
    )
    
    extractor = AsyncPDFExtractor(config)
    
    # Run async function in sync context
    result = asyncio.run(extractor.extract_async())
    
    if not result.success:
        raise PDFExtractionError(result.error_message)
    
    return result.extracted_text or ""

async def main():
    """Main function for running PDF extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract text from PDF using Mistral OCR")
    parser.add_argument(
        "--input", 
        type=str, 
        default="data/raw/hipaa-combined (2).pdf",
        help="Input PDF file path"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="data/clean/hipaa_regulations.txt",
        help="Output text file path"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    def progress_callback(progress: ExtractionProgress):
        """Progress callback for reporting extraction progress."""
        print(f"[{progress.stage.upper()}] {progress.progress_percent:.1f}% - {progress.message}")
    
    try:
        # Create configuration
        config = ExtractionConfig.from_env(
            input_pdf_path=args.input,
            output_text_path=args.output
        )
        
        # Create extractor with progress callback
        extractor = AsyncPDFExtractor(config, progress_callback=progress_callback)
        
        # Extract text
        result = await extractor.extract_async()
        
        # Cleanup
        await extractor.cleanup()
        
        if result.success:
            print(f"\n✅ Success! Extracted {result.pages_processed} pages in {result.processing_time_seconds:.2f}s")
            print(f"📄 Output saved to: {result.output_file_path}")
            print(f"📊 File size: {result.file_size_bytes / (1024*1024):.1f} MB")
            
            # Print first 500 characters as preview
            if result.extracted_text:
                preview = result.extracted_text[:500] + "..." if len(result.extracted_text) > 500 else result.extracted_text
                print(f"\n📖 Preview:\n{preview}")
        else:
            print(f"\n❌ Extraction failed: {result.error_message}")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
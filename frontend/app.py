"""Gradio frontend for HIPAA QA System."""

import asyncio
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import gradio as gr
import httpx
from loguru import logger

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{function}</cyan> | <level>{message}</level>",
    level="INFO"
)

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")
GRADIO_HOST = os.getenv("GRADIO_HOST", "0.0.0.0")
GRADIO_PORT = int(os.getenv("GRADIO_PORT", "7860"))
GRADIO_SHARE = os.getenv("GRADIO_SHARE", "false").lower() == "true"

# HTTP client with timeout
http_client = httpx.AsyncClient(timeout=60.0)


async def ask_question(
    question: str,
    max_chunks: int = 5,
    similarity_threshold: float = 0.4,
    include_sources: bool = True,
) -> Tuple[str, str]:
    """
    Ask a question to the HIPAA QA backend.
    
    Args:
        question: The user's question
        max_chunks: Maximum number of context chunks to retrieve
        similarity_threshold: Minimum similarity threshold
        include_sources: Whether to include source information
        
    Returns:
        Tuple of (answer_text, sources_text)
    """
    if not question.strip():
        return "Please enter a question.", ""
        
    logger.info(f"Processing question: {question[:100]}...")
    
    try:
        # Prepare request
        request_data = {
            "question": question.strip(),
            "max_chunks": max_chunks,
            "similarity_threshold": similarity_threshold,
            "include_metadata": include_sources,
        }
        
        # Make API request
        start_time = time.time()
        response = await http_client.post(
            f"{BACKEND_URL}/qa/ask",
            json=request_data,
            headers={"Content-Type": "application/json"}
        )
        
        response.raise_for_status()
        result = response.json()
        
        processing_time = time.time() - start_time
        
        # Extract answer
        answer = result.get("answer", "No answer provided")
        
        # Format sources if requested
        sources_text = ""
        if include_sources and result.get("sources"):
            sources_text = format_sources(result["sources"])
            
        # Add metadata footer
        metadata = format_metadata(result, processing_time)
        answer_with_metadata = f"{answer}\n\n{metadata}"
        
        logger.info(f"Question processed successfully in {processing_time:.2f}s")
        return answer_with_metadata, sources_text
        
    except httpx.RequestError as e:
        error_msg = f"Connection error: {str(e)}"
        logger.error(error_msg)
        return f"❌ {error_msg}", ""
        
    except httpx.HTTPStatusError as e:
        try:
            error_detail = e.response.json().get("detail", str(e))
        except:
            error_detail = str(e)
        error_msg = f"API error ({e.response.status_code}): {error_detail}"
        logger.error(error_msg)
        return f"❌ {error_msg}", ""
        
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(error_msg)
        return f"❌ {error_msg}", ""


def format_sources(sources: List[Dict]) -> str:
    """Format source references for display."""
    if not sources:
        return "No sources found."
        
    formatted_sources = ["## 📚 Source References\n"]
    
    for i, source in enumerate(sources, 1):
        cfr_citation = source.get("cfr_citation", "Unknown citation")
        section_title = source.get("section_title", "Unknown section")
        similarity_score = source.get("similarity_score", 0.0)
        content_excerpt = source.get("content_excerpt", "")
        
        # Truncate long excerpts
        if len(content_excerpt) > 300:
            content_excerpt = content_excerpt[:300] + "..."
            
        source_text = (
            f"**{i}. {cfr_citation}** - {section_title}\n"
            f"*Similarity: {similarity_score:.3f}*\n\n"
            f"{content_excerpt}\n\n"
            "---\n"
        )
        formatted_sources.append(source_text)
        
    return "".join(formatted_sources)


def format_metadata(result: Dict, processing_time: float) -> str:
    """Format metadata information."""
    chunks_retrieved = result.get("chunks_retrieved", 0)
    confidence_score = result.get("confidence_score")
    model_used = result.get("model_used", "Unknown")
    backend_time = result.get("processing_time_ms", 0)
    
    metadata_parts = [
        "---",
        "### ℹ️ Response Details",
        f"• **Chunks Retrieved:** {chunks_retrieved}",
        f"• **Model:** {model_used}",
        f"• **Processing Time:** {backend_time}ms (backend) + {processing_time*1000:.0f}ms (total)",
    ]
    
    if confidence_score is not None:
        metadata_parts.append(f"• **Confidence:** {confidence_score:.3f}")
        
    return "\n".join(metadata_parts)


def get_backend_health() -> str:
    """Check backend health status."""
    try:
        # Use synchronous httpx client for Gradio compatibility
        with httpx.Client(timeout=10.0) as sync_client:
            response = sync_client.get(f"{BACKEND_URL}/health/")
            response.raise_for_status()
            health_data = response.json()
            
            status = health_data.get("status", "unknown")
            chunks_indexed = health_data.get("chunks_indexed", 0)
            
            if status == "healthy":
                return f"✅ Backend healthy ({chunks_indexed:,} chunks indexed)"
            elif status == "degraded":
                return f"⚠️ Backend degraded ({chunks_indexed:,} chunks indexed)"
            else:
                return f"❌ Backend unhealthy"
                
    except Exception as e:
        return f"❌ Backend unreachable: {str(e)}"


async def get_example_questions() -> List[str]:
    """Get example questions to populate the interface."""
    return [
        "What is a business associate under HIPAA?",
        "What are the requirements for authorization under the Privacy Rule?",
        "What constitutes a breach of protected health information?",
        "What are the administrative safeguards required by the Security Rule?",
        "Who must comply with HIPAA regulations?",
        "What is the definition of protected health information?",
        "What are the penalties for HIPAA violations?",
        "What are the minimum necessary standards?",
        "How should covered entities handle patient access requests?",
        "What encryption standards are required under HIPAA?",
    ]


def create_interface() -> gr.Interface:
    """Create the Gradio interface."""
    
    # Custom CSS for better appearance
    custom_css = """
    .gradio-container {
        max-width: 1200px !important;
    }
    .answer-box {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .sources-box {
        background-color: #f1f3f4;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    """
    
    # Interface function that handles both sync and async
    def interface_fn(question, max_chunks, similarity_threshold, include_sources):
        import asyncio
        
        # Create new event loop if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        # Run the async function
        if loop.is_running():
            # If already in an async context, create a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    lambda: asyncio.run(
                        ask_question(question, max_chunks, similarity_threshold, include_sources)
                    )
                )
                return future.result()
        else:
            return loop.run_until_complete(
                ask_question(question, max_chunks, similarity_threshold, include_sources)
            )
    
    # Create interface
    with gr.Blocks(
        title="HIPAA QA System",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as interface:
        
        # Header
        gr.Markdown(
            """
            # 🏥 HIPAA QA System
            ## Ask questions about HIPAA regulations and get precise answers with citations
            
            This system uses **Retrieval-Augmented Generation (RAG)** to answer questions about HIPAA regulations 
            (45 CFR Parts 160, 162, and 164) with exact citations from the law.
            """
        )
        
        # Health status
        health_status = gr.Markdown("🔄 Checking backend status...")
        
        with gr.Row():
            with gr.Column(scale=2):
                # Question input
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="e.g., What is a business associate under HIPAA?",
                    lines=3,
                    max_lines=5,
                )
                
                # Example questions
                example_questions = gr.Dropdown(
                    label="Example Questions",
                    choices=[
                        "What is a business associate under HIPAA?",
                        "What are the requirements for authorization under the Privacy Rule?",
                        "What constitutes a breach of protected health information?",
                        "What are the administrative safeguards required by the Security Rule?",
                        "Who must comply with HIPAA regulations?",
                        "What is the definition of protected health information?",
                        "What are the penalties for HIPAA violations?",
                        "What are the minimum necessary standards?",
                        "How should covered entities handle patient access requests?",
                        "What encryption standards are required under HIPAA?",
                    ],
                    interactive=True,
                )
                
                # When example is selected, populate the question input
                example_questions.change(
                    fn=lambda x: x,
                    inputs=example_questions,
                    outputs=question_input,
                )
                
            with gr.Column(scale=1):
                # Advanced settings
                with gr.Accordion("Advanced Settings", open=False):
                    max_chunks = gr.Slider(
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1,
                        label="Max Context Chunks",
                        info="Maximum number of regulation sections to retrieve"
                    )
                    
                    similarity_threshold = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.4,
                        step=0.05,
                        label="Similarity Threshold",
                        info="Minimum similarity score for relevant chunks"
                    )
                    
                    include_sources = gr.Checkbox(
                        value=True,
                        label="Show Sources",
                        info="Display source references and citations"
                    )
        
        # Submit button
        submit_btn = gr.Button("Ask Question", variant="primary", size="lg")
        
        # Output areas
        with gr.Row():
            with gr.Column(scale=2):
                answer_output = gr.Markdown(
                    label="Answer",
                    elem_classes=["answer-box"],
                )
                
            with gr.Column(scale=1):
                sources_output = gr.Markdown(
                    label="Sources",
                    elem_classes=["sources-box"],
                    visible=True,
                )
        
        # Event handlers
        submit_btn.click(
            fn=interface_fn,
            inputs=[question_input, max_chunks, similarity_threshold, include_sources],
            outputs=[answer_output, sources_output],
            show_progress=True,
        )
        
        # Also allow Enter key in question input
        question_input.submit(
            fn=interface_fn,
            inputs=[question_input, max_chunks, similarity_threshold, include_sources],
            outputs=[answer_output, sources_output],
            show_progress=True,
        )
        
        # Show/hide sources based on checkbox
        include_sources.change(
            fn=lambda x: gr.update(visible=x),
            inputs=include_sources,
            outputs=sources_output,
        )
        
        # Footer
        gr.Markdown(
            """
            ---
            **About this system:** This HIPAA QA bot uses OpenAI's GPT-4 and text-embedding-3-large models 
            with a PostgreSQL vector database to provide accurate answers with exact citations from the 
            HIPAA regulations. All responses include specific CFR section references.
            
            **Disclaimer:** This system is for informational purposes only and does not constitute legal advice.
            """
        )
        
        # Update health status on load
        interface.load(
            fn=get_backend_health,
            outputs=health_status,
        )
    
    return interface


def main():
    """Main function to run the Gradio app."""
    logger.info("Starting HIPAA QA Frontend...")
    logger.info(f"Backend URL: {BACKEND_URL}")
    logger.info(f"Host: {GRADIO_HOST}, Port: {GRADIO_PORT}")
    
    # Create and launch interface
    interface = create_interface()
    
    interface.launch(
        server_name=GRADIO_HOST,
        server_port=GRADIO_PORT,
        share=GRADIO_SHARE,
        show_error=True,
    )


if __name__ == "__main__":
    main()
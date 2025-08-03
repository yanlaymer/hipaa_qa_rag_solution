"""Question-answering service using RAG with OpenAI."""

import time
from typing import List, Optional, Tuple

import openai
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import Settings, get_settings
from ..database import ChunkRepository, DatabaseManager
from ..schemas import AnswerResponse, ContentType, SourceReference
from .embedding_service import EmbeddingService


class QAService:
    """Service for question-answering using RAG pipeline."""
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        embedding_service: EmbeddingService,
        settings: Optional[Settings] = None,
    ) -> None:
        """Initialize QA service."""
        self.db_manager = db_manager
        self.embedding_service = embedding_service
        self.repository = ChunkRepository(db_manager)
        self.settings = settings or get_settings()
        
        # Configure OpenAI client
        self.client = openai.AsyncOpenAI(
            api_key=self.settings.openai_api_key,
            timeout=self.settings.openai_timeout,
            max_retries=self.settings.openai_max_retries,
        )
        
        self.chat_model = self.settings.openai_chat_model
        logger.info(f"Initialized QA service with model: {self.chat_model}")
        
    async def answer_question(
        self,
        question: str,
        max_chunks: int = 5,
        similarity_threshold: float = 0.7,
        content_types: Optional[List[ContentType]] = None,
        sections: Optional[List[str]] = None,
    ) -> AnswerResponse:
        """
        Answer a question using RAG pipeline.
        
        Args:
            question: User's question about HIPAA
            max_chunks: Maximum number of context chunks to retrieve
            similarity_threshold: Minimum similarity for relevant chunks
            content_types: Optional filter by content types
            sections: Optional filter by specific sections
            
        Returns:
            AnswerResponse with generated answer and source references
        """
        start_time = time.time()
        
        logger.info(f"Processing question: {question[:100]}...")
        
        try:
            # Step 1: Generate question embedding
            question_embedding = await self.embedding_service.embed_text(question)
            
            # Step 2: Retrieve relevant chunks
            chunk_results = await self.repository.similarity_search(
                query_embedding=question_embedding,
                limit=max_chunks,
                similarity_threshold=similarity_threshold,
                content_types=content_types,
                sections=sections,
            )
            
            if not chunk_results:
                logger.warning("No relevant chunks found for question")
                return self._create_no_answer_response(
                    question, start_time, similarity_threshold
                )
                
            logger.info(f"Retrieved {len(chunk_results)} relevant chunks")
            
            # Step 3: Construct context and generate answer
            context = self._build_context(chunk_results)
            answer_text = await self._generate_answer(question, context)
            
            # Step 4: Build source references
            sources = self._build_source_references(chunk_results)
            
            # Step 5: Calculate processing time and build response
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            # Calculate confidence score based on similarity scores
            confidence_score = self._calculate_confidence(chunk_results)
            
            return AnswerResponse(
                question=question,
                answer=answer_text,
                sources=sources,
                confidence_score=confidence_score,
                processing_time_ms=processing_time_ms,
                model_used=self.chat_model,
                chunks_retrieved=len(chunk_results),
                metadata={
                    "similarity_threshold": similarity_threshold,
                    "embedding_model": self.embedding_service.model,
                    "max_chunks_requested": max_chunks,
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            return AnswerResponse(
                question=question,
                answer=f"I apologize, but I encountered an error processing your question: {str(e)}",
                sources=[],
                confidence_score=0.0,
                processing_time_ms=processing_time_ms,
                model_used=self.chat_model,
                chunks_retrieved=0,
                metadata={"error": str(e)}
            )
            
    def _build_context(self, chunk_results: List[Tuple]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []
        
        for i, (chunk, similarity_score) in enumerate(chunk_results, 1):
            # Format each chunk with clear section identification
            section_ref = chunk.cfr_citation or chunk.full_reference
            context_part = (
                f"[Source {i}: {section_ref}]\n"
                f"{chunk.content}\n"
            )
            context_parts.append(context_part)
            
        return "\n".join(context_parts)
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError)),
    )
    async def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using OpenAI chat completion."""
        
        system_prompt = self._get_system_prompt()
        user_prompt = self._build_user_prompt(context, question)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,  # Low temperature for consistent, factual responses
                max_tokens=2000,  # Reasonable limit for responses
                top_p=0.9,
            )
            
            if not response.choices:
                raise ValueError("No response generated by the model")
                
            answer = response.choices[0].message.content
            
            if not answer:
                raise ValueError("Empty response generated by the model")
                
            logger.debug(f"Generated answer length: {len(answer)} characters")
            return answer.strip()
            
        except openai.OpenAIError as e:
            logger.error(f"OpenAI API error during answer generation: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during answer generation: {e}")
            raise
            
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the AI assistant."""
        return """You are a HIPAA expert AI assistant with access to the complete text of the HIPAA regulations (45 CFR Parts 160, 162, and 164). Your role is to provide accurate, precise answers to questions about HIPAA compliance and regulations.

CRITICAL INSTRUCTIONS:
1. Answer questions using ONLY the information provided in the context sources
2. Quote the exact relevant text from the regulations when possible
3. Always cite the specific CFR section number (e.g., "45 CFR § 164.502") for each factual statement
4. If the answer requires information from multiple sections, cite each one
5. If the provided context doesn't contain sufficient information to answer the question, clearly state that you don't have enough information
6. Do not provide information outside the given sources
7. Use clear, professional language appropriate for legal/regulatory context
8. Structure your answer logically with proper citations throughout

FORMAT FOR CITATIONS:
- Use format: "According to 45 CFR § [section], [quoted text or paraphrased content]"
- Place citations immediately after the relevant statement
- If quoting directly, use quotation marks around the regulatory text

ANSWER STRUCTURE:
1. Direct answer to the question
2. Supporting quotes from the regulation with proper citations
3. Additional relevant context if helpful
4. Clear indication if information is incomplete"""

    def _build_user_prompt(self, context: str, question: str) -> str:
        """Build the user prompt with context and question."""
        return f"""Based on the following HIPAA regulation excerpts, please answer the user's question with exact citations.

REGULATION CONTEXT:
{context}

USER QUESTION: {question}

Please provide a comprehensive answer with proper CFR citations."""
        
    def _build_source_references(self, chunk_results: List[Tuple]) -> List[SourceReference]:
        """Build source reference objects from chunk results."""
        sources = []
        
        for chunk, similarity_score in chunk_results:
            # Create content excerpt (first 200 chars + ellipsis if longer)
            content_excerpt = chunk.content
            if len(content_excerpt) > 200:
                content_excerpt = content_excerpt[:200] + "..."
                
            source = SourceReference(
                section_id=chunk.section_id,
                cfr_citation=chunk.cfr_citation or chunk.full_reference,
                section_title=chunk.section_title,
                content_excerpt=content_excerpt,
                similarity_score=round(similarity_score, 3),
                chunk_id=chunk.chunk_id,
            )
            sources.append(source)
            
        return sources
        
    def _calculate_confidence(self, chunk_results: List[Tuple]) -> float:
        """Calculate confidence score based on similarity scores."""
        if not chunk_results:
            return 0.0
            
        # Use the highest similarity score as primary confidence indicator
        max_similarity = max(similarity for _, similarity in chunk_results)
        
        # Apply additional factors
        num_chunks = len(chunk_results)
        
        # Base confidence on max similarity
        confidence = max_similarity
        
        # Boost confidence if we have multiple relevant chunks
        if num_chunks >= 3:
            confidence = min(confidence * 1.1, 1.0)
        elif num_chunks >= 2:
            confidence = min(confidence * 1.05, 1.0)
            
        return round(confidence, 3)
        
    def _create_no_answer_response(
        self,
        question: str,
        start_time: float,
        similarity_threshold: float
    ) -> AnswerResponse:
        """Create response when no relevant chunks are found."""
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        no_answer_text = (
            "I apologize, but I couldn't find relevant information in the HIPAA "
            "regulations to answer your question. This could mean:\n\n"
            "1. The question is outside the scope of HIPAA regulations (Parts 160, 162, 164)\n"
            "2. The question might need to be rephrased more specifically\n"
            "3. The information might be in a section not well-matched by the search\n\n"
            "Please try rephrasing your question or asking about a specific HIPAA topic "
            "like privacy rules, security standards, breach notification, or covered entities."
        )
        
        return AnswerResponse(
            question=question,
            answer=no_answer_text,
            sources=[],
            confidence_score=0.0,
            processing_time_ms=processing_time_ms,
            model_used=self.chat_model,
            chunks_retrieved=0,
            metadata={
                "reason": "no_relevant_chunks",
                "similarity_threshold": similarity_threshold,
            }
        )
        
    async def validate_chat_api_access(self) -> bool:
        """
        Validate that the OpenAI Chat API is accessible.
        
        Returns:
            True if API is accessible, False otherwise
        """
        try:
            test_response = await self.client.chat.completions.create(
                model=self.chat_model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
                temperature=0.1,
            )
            
            if test_response.choices and test_response.choices[0].message.content:
                logger.info("OpenAI Chat API validation successful")
                return True
            else:
                logger.error("Chat API validation failed: no response content")
                return False
                
        except Exception as e:
            logger.error(f"OpenAI Chat API validation failed: {e}")
            return False
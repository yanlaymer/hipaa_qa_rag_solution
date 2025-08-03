"""
Enhanced QA Service for 100% Accuracy

This enhanced QA service addresses specific issues identified in testing:
1. Better domain-aware retrieval (Privacy vs Security)
2. Improved system prompts for accurate responses
3. Enhanced context construction for precise citations
4. Better similarity thresholding
"""

import time
import logging
from typing import List, Optional, Tuple, Dict, Any

import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from ..config import Settings
from ..database import DatabaseManager
from ..schemas import AnswerResponse, SourceReference, ContentType
from .embedding_service import EmbeddingService


logger = logging.getLogger(__name__)


class EnhancedQAService:
    """Enhanced QA Service targeting 100% accuracy on HIPAA questions."""
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        embedding_service: EmbeddingService,
        settings: Optional[Settings] = None,
    ) -> None:
        self.db_manager = db_manager
        self.embedding_service = embedding_service
        self.settings = settings or Settings()
        self.repository = db_manager.get_repository()
        
        # OpenAI client
        self.client = openai.AsyncOpenAI(api_key=self.settings.openai_api_key)
        self.chat_model = self.settings.openai_chat_model
        
        # Enhanced retrieval settings
        self.domain_keywords = {
            'privacy': ['privacy', 'protected health information', 'phi', 'uses and disclosures', 
                       'authorization', 'minimum necessary', 'disclosure', 'family members'],
            'security': ['security', 'safeguards', 'access control', 'encryption', 'integrity', 
                        'transmission', 'audit', 'technical safeguards'],
            'penalties': ['penalties', 'civil money penalty', 'violation', 'fine', 'sanctions'],
            'general': ['covered entities', 'business associates', 'applicability', 'definitions'],
            'transactions': ['transactions', 'standards', 'code sets', 'identifiers']
        }
    
    async def answer_question(
        self,
        question: str,
        max_chunks: int = 8,  # Increased for better coverage
        similarity_threshold: float = 0.1,  # Lower threshold for better recall
        content_types: Optional[List[ContentType]] = None,
        sections: Optional[List[str]] = None,
    ) -> AnswerResponse:
        """Answer a question with enhanced accuracy."""
        start_time = time.time()
        
        logger.info(f"🔍 Processing enhanced question: {question[:100]}...")
        
        try:
            # Step 1: Detect question domain for better retrieval
            question_domain = self._detect_question_domain(question)
            logger.info(f"🏷️ Detected question domain: {question_domain}")
            
            # Step 2: Generate question embedding
            question_embedding = await self.embedding_service.embed_text(question)
            
            # Step 3: Enhanced retrieval with domain awareness
            chunk_results = await self._enhanced_retrieval(
                question=question,
                question_embedding=question_embedding,
                question_domain=question_domain,
                max_chunks=max_chunks,
                similarity_threshold=similarity_threshold,
                content_types=content_types,
                sections=sections,
            )
            
            if not chunk_results:
                logger.warning("No relevant chunks found for question")
                return self._create_no_answer_response(
                    question, start_time, similarity_threshold
                )
                
            logger.info(f"📋 Retrieved {len(chunk_results)} relevant chunks from domain: {question_domain}")
            
            # Step 4: Enhanced context construction
            context = self._build_enhanced_context(chunk_results, question_domain)
            
            # Step 5: Generate answer with enhanced prompts
            answer_text = await self._generate_enhanced_answer(question, context, question_domain)
            
            # Step 6: Build enhanced source references
            sources = self._build_enhanced_source_references(chunk_results)
            
            # Step 7: Calculate processing time and build response
            processing_time_ms = int((time.time() - start_time) * 1000)
            confidence_score = self._calculate_enhanced_confidence(chunk_results, question_domain)
            
            return AnswerResponse(
                question=question,
                answer=answer_text,
                sources=sources,
                confidence_score=confidence_score,
                processing_time_ms=processing_time_ms,
                model_used=self.chat_model,
                chunks_retrieved=len(chunk_results),
                metadata={
                    "question_domain": question_domain,
                    "similarity_threshold": similarity_threshold,
                    "enhancement_version": "1.0"
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Error in enhanced QA processing: {e}")
            raise
    
    def _detect_question_domain(self, question: str) -> str:
        """Detect the regulatory domain of the question."""
        question_lower = question.lower()
        domain_scores = {}
        
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            domain_scores[domain] = score
        
        # Special domain detection rules
        if any(word in question_lower for word in ['privacy', 'protected health information', 'phi', 'disclosure', 'family']):
            domain_scores['privacy'] = domain_scores.get('privacy', 0) + 5
            
        if any(word in question_lower for word in ['security', 'encryption', 'safeguards', 'access control']):
            domain_scores['security'] = domain_scores.get('security', 0) + 5
            
        if any(word in question_lower for word in ['penalty', 'penalties', 'fine', 'civil money', 'violation']):
            domain_scores['penalties'] = domain_scores.get('penalties', 0) + 5
            
        if any(word in question_lower for word in ['covered entities', 'business associate', 'applicability']):
            domain_scores['general'] = domain_scores.get('general', 0) + 3
        
        # Get domain with highest score
        best_domain = max(domain_scores.items(), key=lambda x: x[1])
        return best_domain[0] if best_domain[1] > 0 else 'general'
    
    async def _enhanced_retrieval(
        self,
        question: str,
        question_embedding: List[float],
        question_domain: str,
        max_chunks: int,
        similarity_threshold: float,
        content_types: Optional[List[ContentType]] = None,
        sections: Optional[List[str]] = None,
    ) -> List[Tuple]:
        """Enhanced retrieval with domain awareness."""
        
        # First, try domain-specific retrieval
        domain_chunks = await self._retrieve_by_domain(
            question_embedding, question_domain, max_chunks // 2, similarity_threshold
        )
        
        # Then, general retrieval for additional context
        general_chunks = await self.repository.similarity_search(
            query_embedding=question_embedding,
            limit=max_chunks,
            similarity_threshold=similarity_threshold,
            content_types=content_types,
            sections=sections,
        )
        
        # Combine and deduplicate
        all_chunks = domain_chunks + general_chunks
        seen_chunk_ids = set()
        unique_chunks = []
        
        for chunk in all_chunks:
            chunk_id = chunk[0]  # Assuming first element is chunk_id
            if chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                unique_chunks.append(chunk)
        
        # Sort by similarity score (descending)
        unique_chunks.sort(key=lambda x: x[-1], reverse=True)  # Last element is similarity
        
        return unique_chunks[:max_chunks]
    
    async def _retrieve_by_domain(
        self, 
        query_embedding: List[float], 
        domain: str, 
        limit: int, 
        threshold: float
    ) -> List[Tuple]:
        """Retrieve chunks filtered by regulation domain."""
        # This would require database schema enhancement to filter by domain
        # For now, fall back to regular retrieval
        return await self.repository.similarity_search(
            query_embedding=query_embedding,
            limit=limit,
            similarity_threshold=threshold
        )
    
    def _build_enhanced_context(self, chunk_results: List[Tuple], question_domain: str) -> str:
        """Build enhanced context with domain awareness."""
        context_parts = []
        
        # Group chunks by domain and section
        domain_chunks = {}
        for i, chunk in enumerate(chunk_results):
            chunk_id, content, section_id, cfr_citation, references_json, key_terms_json, similarity = chunk
            
            # Determine chunk domain (would be better with actual domain metadata)
            if 'privacy' in question_domain and ('164.5' in section_id or 'privacy' in content.lower()):
                chunk_domain = 'privacy'
            elif 'security' in question_domain and ('164.3' in section_id or 'security' in content.lower()):
                chunk_domain = 'security'
            else:
                chunk_domain = 'general'
            
            if chunk_domain not in domain_chunks:
                domain_chunks[chunk_domain] = []
            domain_chunks[chunk_domain].append((i + 1, chunk))
        
        # Build context with domain grouping
        for domain, chunks in domain_chunks.items():
            if chunks:
                context_parts.append(f"=== {domain.upper()} REGULATIONS ===")
                for source_num, chunk in chunks:
                    chunk_id, content, section_id, cfr_citation, references_json, key_terms_json, similarity = chunk
                    context_part = f"[Source {source_num} - {cfr_citation or section_id}]\n{content}"
                    context_parts.append(context_part)
                context_parts.append("")  # Empty line between domains
        
        return "\n".join(context_parts)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError)),
    )
    async def _generate_enhanced_answer(self, question: str, context: str, question_domain: str) -> str:
        """Generate answer with enhanced prompts."""
        
        system_prompt = self._get_enhanced_system_prompt(question_domain)
        user_prompt = self._build_enhanced_user_prompt(context, question, question_domain)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,  # Zero temperature for maximum consistency
                max_tokens=2500,  # Increased for comprehensive answers
                top_p=1.0,
            )
            
            if not response.choices:
                raise ValueError("No response generated by the model")
                
            answer = response.choices[0].message.content
            
            if not answer:
                raise ValueError("Empty response generated by the model")
                
            logger.debug(f"Generated enhanced answer length: {len(answer)} characters")
            return answer.strip()
            
        except openai.OpenAIError as e:
            logger.error(f"OpenAI API error during enhanced answer generation: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during enhanced answer generation: {e}")
            raise
    
    def _get_enhanced_system_prompt(self, question_domain: str) -> str:
        """Get enhanced system prompt based on question domain."""
        
        base_prompt = """You are a HIPAA expert AI assistant with access to the complete text of the HIPAA regulations (45 CFR Parts 160, 162, and 164). Your role is to provide accurate, precise answers to questions about HIPAA compliance and regulations.

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

        domain_specific_guidance = {
            'privacy': """
PRIVACY RULE SPECIFIC GUIDANCE:
- Focus on Part 164, Subpart E (Privacy of Individually Identifiable Health Information)
- Key sections: 164.500-164.534
- Pay special attention to uses and disclosures (164.502), minimum necessary (164.502(b)), authorizations (164.508), and permitted disclosures (164.510-164.512)
- Distinguish between Privacy Rule (Subpart E) and Security Rule (Subpart C)""",
            
            'security': """
SECURITY RULE SPECIFIC GUIDANCE:
- Focus on Part 164, Subpart C (Security Standards for Electronic Protected Health Information)
- Key sections: 164.302-164.318
- Pay attention to administrative (164.308), physical (164.310), and technical safeguards (164.312)
- Distinguish between required and addressable implementation specifications""",
            
            'penalties': """
PENALTIES SPECIFIC GUIDANCE:
- Focus on Part 160, Subpart D (Civil Money Penalties)
- Key sections: 160.400-160.426
- Pay attention to penalty amounts (160.404), factors considered (160.408), and violation categories
- Include specific dollar amounts and penalty tiers when relevant""",
            
            'general': """
GENERAL PROVISIONS GUIDANCE:
- Focus on Part 160 (General Administrative Requirements)
- Key sections: definitions (160.103), applicability (160.102), covered entities and business associates
- Provide clear distinctions between different entity types and their obligations"""
        }
        
        domain_guidance = domain_specific_guidance.get(question_domain, "")
        
        return base_prompt + "\n" + domain_guidance
    
    def _build_enhanced_user_prompt(self, context: str, question: str, question_domain: str) -> str:
        """Build enhanced user prompt with domain context."""
        
        domain_instructions = {
            'privacy': "Focus on privacy-related provisions and clearly distinguish between Privacy Rule (164.E) and Security Rule (164.C).",
            'security': "Focus on security safeguards and technical requirements for electronic PHI protection.",
            'penalties': "Provide specific penalty amounts and cite the exact penalty calculation provisions.",
            'general': "Provide clear definitions and explain applicability to different entity types."
        }
        
        instruction = domain_instructions.get(question_domain, "Provide a comprehensive answer with exact citations.")
        
        return f"""Based on the following HIPAA regulation excerpts, please answer the user's question with exact citations.

{instruction}

REGULATION CONTEXT:
{context}

USER QUESTION: {question}

Please provide a comprehensive answer with proper CFR citations, focusing on the {question_domain} domain."""
    
    def _build_enhanced_source_references(self, chunk_results: List[Tuple]) -> List[SourceReference]:
        """Build enhanced source references."""
        sources = []
        
        for i, chunk in enumerate(chunk_results):
            chunk_id, content, section_id, cfr_citation, references_json, key_terms_json, similarity = chunk
            
            sources.append(SourceReference(
                source_id=f"source_{i + 1}",
                title=cfr_citation or f"45 CFR § {section_id}",
                section_reference=cfr_citation or f"45 CFR § {section_id}",
                content_preview=content[:200] + "..." if len(content) > 200 else content,
                similarity_score=float(similarity),
                metadata={
                    "chunk_id": chunk_id,
                    "section_id": section_id,
                    "full_content_length": len(content)
                }
            ))
        
        return sources
    
    def _calculate_enhanced_confidence(self, chunk_results: List[Tuple], question_domain: str) -> float:
        """Calculate enhanced confidence score."""
        if not chunk_results:
            return 0.0
        
        # Base confidence from similarity scores
        similarities = [float(chunk[-1]) for chunk in chunk_results]  # Last element is similarity
        avg_similarity = sum(similarities) / len(similarities)
        
        # Boost confidence for domain-specific matches
        domain_boost = 0.0
        for chunk in chunk_results:
            content = chunk[1].lower()
            if question_domain == 'privacy' and any(word in content for word in ['privacy', 'protected health information', 'disclosure']):
                domain_boost += 0.1
            elif question_domain == 'security' and any(word in content for word in ['security', 'safeguards', 'encryption']):
                domain_boost += 0.1
            elif question_domain == 'penalties' and any(word in content for word in ['penalty', 'civil money', 'violation']):
                domain_boost += 0.1
        
        # Confidence calculation
        confidence = min(avg_similarity + domain_boost, 1.0)
        
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
                "enhancement_version": "1.0"
            }
        )
"""OpenAI embedding service for text vectorization."""

import asyncio
from typing import List, Optional

import openai
from loguru import logger
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..config import Settings, get_settings


class EmbeddingService:
    """Service for generating embeddings using OpenAI."""
    
    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize embedding service."""
        self.settings = settings or get_settings()
        
        # Configure OpenAI client
        self.client = openai.AsyncOpenAI(
            api_key=self.settings.openai_api_key,
            timeout=self.settings.openai_timeout,
            max_retries=self.settings.openai_max_retries,
        )
        
        self.model = self.settings.openai_embedding_model
        logger.info(f"Initialized embedding service with model: {self.model}")
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError)),
    )
    async def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of embedding values
            
        Raises:
            openai.OpenAIError: If API call fails after retries
        """
        if not text.strip():
            raise ValueError("Text cannot be empty")
            
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text.strip(),
                encoding_format="float"
            )
            
            embedding = response.data[0].embedding
            
            # Validate embedding dimension
            expected_dim = self.settings.embedding_dimension
            if len(embedding) != expected_dim:
                raise ValueError(
                    f"Unexpected embedding dimension: got {len(embedding)}, "
                    f"expected {expected_dim}"
                )
                
            logger.debug(f"Generated embedding for text (length: {len(text)})")
            return embedding
            
        except openai.OpenAIError as e:
            logger.error(f"OpenAI API error during embedding: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during embedding: {e}")
            raise
            
    async def embed_texts_batch(
        self, 
        texts: List[str],
        batch_size: int = 100
    ) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process per API call
            
        Returns:
            List of embeddings corresponding to input texts
            
        Raises:
            openai.OpenAIError: If API calls fail after retries
        """
        if not texts:
            return []
            
        # Filter out empty texts and track indices
        valid_texts = []
        valid_indices = []
        for i, text in enumerate(texts):
            if text.strip():
                valid_texts.append(text.strip())
                valid_indices.append(i)
            else:
                logger.warning(f"Skipping empty text at index {i}")
                
        if not valid_texts:
            raise ValueError("No valid texts to embed")
            
        logger.info(f"Embedding {len(valid_texts)} texts in batches of {batch_size}")
        
        all_embeddings = []
        
        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i:i + batch_size]
            
            try:
                embeddings = await self._embed_batch(batch)
                all_embeddings.extend(embeddings)
                
                logger.info(
                    f"Processed batch {i//batch_size + 1}/"
                    f"{(len(valid_texts)-1)//batch_size + 1}"
                )
                
                # Rate limiting: small delay between batches
                if i + batch_size < len(valid_texts):
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Failed to process batch {i//batch_size + 1}: {e}")
                raise
                
        # Reconstruct full results list with None for empty texts
        results = [None] * len(texts)  # type: ignore
        for i, embedding in enumerate(all_embeddings):
            original_index = valid_indices[i]
            results[original_index] = embedding
            
        logger.info(f"Successfully embedded {len(all_embeddings)} texts")
        return results  # type: ignore
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APITimeoutError)),
    )
    async def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Internal method to embed a batch of texts."""
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=texts,
                encoding_format="float"
            )
            
            embeddings = [data.embedding for data in response.data]
            
            # Validate all embeddings have correct dimension
            expected_dim = self.settings.embedding_dimension
            for i, embedding in enumerate(embeddings):
                if len(embedding) != expected_dim:
                    raise ValueError(
                        f"Unexpected embedding dimension for text {i}: "
                        f"got {len(embedding)}, expected {expected_dim}"
                    )
                    
            return embeddings
            
        except openai.OpenAIError as e:
            logger.error(f"OpenAI API error during batch embedding: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during batch embedding: {e}")
            raise
            
    async def validate_api_access(self) -> bool:
        """
        Validate that the OpenAI API is accessible with current credentials.
        
        Returns:
            True if API is accessible, False otherwise
        """
        try:
            # Test with a simple embedding request
            test_response = await self.client.embeddings.create(
                model=self.model,
                input="test",
                encoding_format="float"
            )
            
            if test_response.data and len(test_response.data) > 0:
                embedding = test_response.data[0].embedding
                expected_dim = self.settings.embedding_dimension
                
                if len(embedding) == expected_dim:
                    logger.info("OpenAI embedding API validation successful")
                    return True
                else:
                    logger.error(
                        f"API validation failed: unexpected embedding dimension "
                        f"{len(embedding)}, expected {expected_dim}"
                    )
                    return False
            else:
                logger.error("API validation failed: no embedding data returned")
                return False
                
        except Exception as e:
            logger.error(f"OpenAI API validation failed: {e}")
            return False
            
    def get_model_info(self) -> dict:
        """Get information about the current embedding model."""
        return {
            "model": self.model,
            "dimension": self.settings.embedding_dimension,
            "max_tokens": 8191,  # text-embedding-3-large limit
            "cost_per_1k_tokens": 0.00013,  # As of latest pricing
        }
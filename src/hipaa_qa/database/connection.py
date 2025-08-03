"""Database connection management for PostgreSQL with pgvector."""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncGenerator, Optional

import asyncpg
from loguru import logger
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from ..config import Settings, get_settings


class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, settings: Optional[Settings] = None) -> None:
        """Initialize database manager."""
        self.settings = settings or get_settings()
        self._engine: Optional[AsyncEngine] = None
        self._session_factory: Optional[async_sessionmaker[AsyncSession]] = None
        self._connection_pool: Optional[asyncpg.Pool] = None
        
    @property
    def engine(self) -> AsyncEngine:
        """Get or create the SQLAlchemy async engine."""
        if self._engine is None:
            self._engine = create_async_engine(
                self.settings.database_url,
                echo=self.settings.api_debug,
                poolclass=NullPool,  # Use asyncpg pool instead
                pool_pre_ping=True,
                future=True,
            )
            logger.info("Created SQLAlchemy async engine")
        return self._engine
        
    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        """Get or create the session factory."""
        if self._session_factory is None:
            self._session_factory = async_sessionmaker(
                bind=self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )
            logger.info("Created SQLAlchemy session factory")
        return self._session_factory
        
    async def get_connection_pool(self) -> asyncpg.Pool:
        """Get or create the asyncpg connection pool."""
        if self._connection_pool is None:
            self._connection_pool = await asyncpg.create_pool(
                host=self.settings.db_host,
                port=self.settings.db_port,
                user=self.settings.db_user,
                password=self.settings.db_password,
                database=self.settings.db_name,
                min_size=1,
                max_size=self.settings.db_pool_size,
                command_timeout=60,
                server_settings={
                    'application_name': 'hipaa_qa_system',
                },
            )
            logger.info(
                f"Created asyncpg connection pool "
                f"(size: {self.settings.db_pool_size})"
            )
        return self._connection_pool
        
    async def initialize_database(self) -> None:
        """Initialize database with pgvector extension and tables."""
        logger.info("Initializing database...")
        
        try:
            pool = await self.get_connection_pool()
            async with pool.acquire() as conn:
                # Enable pgvector extension
                await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
                logger.info("Enabled pgvector extension")
                
                # Import and create tables
                from .models import Base
                async with self.engine.begin() as conn:
                    await conn.run_sync(Base.metadata.create_all)
                logger.info("Created database tables")
                
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise
            
    async def check_connection(self) -> bool:
        """Check if database connection is healthy."""
        try:
            pool = await self.get_connection_pool()
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
            
    async def get_chunks_count(self) -> int:
        """Get the total number of document chunks in the database."""
        try:
            pool = await self.get_connection_pool()
            async with pool.acquire() as conn:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM document_chunks"
                )
                return count or 0
        except Exception as e:
            logger.error(f"Failed to get chunks count: {e}")
            return 0
            
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get an async database session."""
        async with self.session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
                
    @asynccontextmanager
    async def get_connection(self) -> AsyncGenerator[asyncpg.Connection, None]:
        """Get a raw asyncpg connection for vector operations."""
        pool = await self.get_connection_pool()
        async with pool.acquire() as conn:
            yield conn
            
    async def close(self) -> None:
        """Close all database connections."""
        if self._connection_pool:
            await self._connection_pool.close()
            logger.info("Closed asyncpg connection pool")
            
        if self._engine:
            await self._engine.dispose()
            logger.info("Disposed SQLAlchemy engine")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


async def wait_for_database(
    max_retries: int = 30,
    retry_interval: float = 1.0,
    settings: Optional[Settings] = None,
) -> bool:
    """Wait for database to become available."""
    settings = settings or get_settings()
    
    for attempt in range(max_retries):
        try:
            # Try to connect using asyncpg directly
            conn = await asyncpg.connect(
                host=settings.db_host,
                port=settings.db_port,
                user=settings.db_user,
                password=settings.db_password,
                database=settings.db_name,
                timeout=5.0,
            )
            await conn.fetchval("SELECT 1")
            await conn.close()
            logger.info(f"Database connection successful after {attempt + 1} attempts")
            return True
            
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Database connection attempt {attempt + 1} failed: {e}. "
                    f"Retrying in {retry_interval}s..."
                )
                await asyncio.sleep(retry_interval)
            else:
                logger.error(f"Database connection failed after {max_retries} attempts: {e}")
                return False
                
    return False
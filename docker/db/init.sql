-- Database initialization script for HIPAA QA System

-- Create vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify extension installation
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';

-- Create database user if not exists (optional, as postgres user is default)
-- You can customize this based on your needs

-- Log successful initialization
\echo 'pgvector extension installed successfully'
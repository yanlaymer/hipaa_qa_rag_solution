#!/bin/bash

# Development script to run HIPAA QA System locally

set -e

echo "🏥 Starting HIPAA QA System in development mode..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Please copy env.example to .env and configure your settings."
    exit 1
fi

# Check if OpenAI API key is set
if ! grep -q "^OPENAI_API_KEY=sk-" .env; then
    echo "❌ OPENAI_API_KEY not set in .env file. Please add your OpenAI API key."
    exit 1
fi

# Create logs directory
mkdir -p logs

# Build and start services
echo "🐳 Building Docker images..."
docker-compose build

echo "📊 Starting database..."
docker-compose up -d db

echo "⏳ Waiting for database to be ready..."
sleep 10

# Check if data needs to be ingested
echo "📥 Checking if data ingestion is needed..."
if docker-compose exec -T db psql -U postgres -d hipaa_qa -c "SELECT COUNT(*) FROM document_chunks;" 2>/dev/null | grep -q " 0"; then
    echo "📦 No data found. Running ingestion..."
    docker-compose run --rm backend python /app/scripts/ingest_enhanced_data.py
else
    echo "✅ Data already exists in database"
fi

echo "🚀 Starting all services..."
docker-compose up -d

echo "⏳ Waiting for services to be ready..."
sleep 15

echo ""
echo "🎉 HIPAA QA System is running!"
echo ""
echo "📊 Services:"
echo "  - Backend API: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - Frontend UI: http://localhost:7860"
echo "  - Database: localhost:5432"
echo ""
echo "🌐 Public URL (via Cloudflare Tunnel):"
echo "  Check the cloudflared container logs for the public URL:"
echo "  docker-compose logs cloudflared | grep trycloudflare.com"
echo ""
echo "📊 Useful commands:"
echo "  - View logs: docker-compose logs -f [service]"
echo "  - Stop system: docker-compose down"
echo "  - Restart: docker-compose restart [service]"
echo "  - Health check: curl http://localhost:8000/health"
echo ""

# Show the Cloudflare tunnel URL
echo "🔗 Getting Cloudflare tunnel URL..."
sleep 5
docker-compose logs cloudflared 2>/dev/null | grep -o "https://[a-zA-Z0-9-]*\.trycloudflare\.com" | tail -1 | xargs -I {} echo "  Public URL: {}"
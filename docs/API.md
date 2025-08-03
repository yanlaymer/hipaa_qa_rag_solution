# 🔌 API Documentation

The HIPAA QA System provides a comprehensive REST API built with FastAPI for programmatic access to the question-answering capabilities.

## 🌟 Base URL

- **Local**: `http://localhost/api` (through nginx)
- **Direct Backend**: `http://localhost:8000` (development)
- **Production**: `https://your-domain.com/api`

## 📖 Interactive Documentation

- **Swagger UI**: `http://localhost/api/docs`
- **ReDoc**: `http://localhost/api/redoc`
- **OpenAPI JSON**: `http://localhost/api/openapi.json`

## 🔍 Endpoints

### Question & Answer

#### POST `/qa/ask`

Ask a question about HIPAA regulations and receive an answer with citations.

**Request Body:**
```json
{
  "question": "string (required)",
  "similarity_threshold": "float (optional, default: 0.4)",
  "max_chunks": "integer (optional, default: 5)",
  "include_metadata": "boolean (optional, default: true)"
}
```

**Response:**
```json
{
  "answer": "string",
  "sources": [
    {
      "chunk_id": "integer",
      "section": "string",
      "part": "string", 
      "similarity_score": "float",
      "content": "string"
    }
  ],
  "metadata": {
    "chunks_retrieved": "integer",
    "processing_time_ms": "float",
    "model_used": "string",
    "confidence_score": "float"
  }
}
```

**Example:**
```bash
curl -X POST "http://localhost/api/qa/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is a business associate under HIPAA?",
    "similarity_threshold": 0.4,
    "max_chunks": 5
  }'
```

**Response Example:**
```json
{
  "answer": "Under HIPAA, a business associate is a person or organization that performs functions or activities on behalf of a covered entity that involve the use or disclosure of protected health information (PHI)...",
  "sources": [
    {
      "chunk_id": 123,
      "section": "§ 160.103",
      "part": "45 CFR Part 160",
      "similarity_score": 0.89,
      "content": "Business associate means, with respect to a covered entity, a person who..."
    }
  ],
  "metadata": {
    "chunks_retrieved": 5,
    "processing_time_ms": 1250.5,
    "model_used": "gpt-4-turbo",
    "confidence_score": 0.92
  }
}
```

### Health Checks

#### GET `/health/`

Basic health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-03T14:30:00Z",
  "version": "0.1.0"
}
```

#### GET `/health/live`

Liveness probe for container orchestration.

**Response:**
```json
{
  "status": "healthy"
}
```

#### GET `/health/ready`

Readiness probe checking all dependencies.

**Response:**
```json
{
  "status": "healthy",
  "database_connected": true,
  "openai_accessible": true,
  "chunks_indexed": 882
}
```

#### GET `/health/db`

Database-specific health check.

**Response:**
```json
{
  "status": "healthy",
  "database_connected": true,
  "chunks_count": 882,
  "connection_pool": {
    "active": 2,
    "idle": 8,
    "total": 10
  }
}
```

### System Information

#### GET `/models`

Get information about available AI models.

**Response:**
```json
{
  "chat_model": "gpt-4-turbo",
  "embedding_model": "text-embedding-3-large",
  "embedding_dimension": 3072
}
```

#### GET `/stats`

Get system statistics and metrics.

**Response:**
```json
{
  "total_chunks": 882,
  "total_questions_processed": 1547,
  "average_response_time_ms": 1250.5,
  "uptime_seconds": 86400,
  "memory_usage_mb": 512.3
}
```

## 🔐 Authentication

Currently, the API does not require authentication. For production deployments, consider implementing:

- API Key authentication
- OAuth 2.0 / JWT tokens
- IP allowlisting
- Rate limiting (already implemented)

### Future Authentication Example

```bash
# With API key (future implementation)
curl -X POST "http://localhost/api/qa/ask" \
  -H "Authorization: Bearer your_api_key" \
  -H "Content-Type: application/json" \
  -d '{"question": "Your question here"}'
```

## 📊 Rate Limiting

The API implements rate limiting through nginx:

- **API Endpoints**: 10 requests/second per IP
- **Burst Capacity**: 20 requests
- **Frontend**: 30 requests/second per IP

Rate limit headers:
```
X-RateLimit-Limit: 10
X-RateLimit-Remaining: 9
X-RateLimit-Reset: 1625097600
```

## 🐍 Python Client

### Synchronous Client

```python
import requests

class HIPAAQAClient:
    def __init__(self, base_url: str = "http://localhost/api"):
        self.base_url = base_url
    
    def ask_question(
        self, 
        question: str,
        similarity_threshold: float = 0.4,
        max_chunks: int = 5
    ) -> dict:
        response = requests.post(
            f"{self.base_url}/qa/ask",
            json={
                "question": question,
                "similarity_threshold": similarity_threshold,
                "max_chunks": max_chunks
            }
        )
        response.raise_for_status()
        return response.json()
    
    def health_check(self) -> dict:
        response = requests.get(f"{self.base_url}/health/")
        response.raise_for_status()
        return response.json()

# Usage
client = HIPAAQAClient()
result = client.ask_question("What is a covered entity?")
print(result["answer"])
```

### Asynchronous Client

```python
import asyncio
import httpx

class AsyncHIPAAQAClient:
    def __init__(self, base_url: str = "http://localhost/api"):
        self.base_url = base_url
    
    async def ask_question(
        self, 
        question: str,
        similarity_threshold: float = 0.4,
        max_chunks: int = 5
    ) -> dict:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/qa/ask",
                json={
                    "question": question,
                    "similarity_threshold": similarity_threshold,
                    "max_chunks": max_chunks
                }
            )
            response.raise_for_status()
            return response.json()
    
    async def health_check(self) -> dict:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/health/")
            response.raise_for_status()
            return response.json()

# Usage
async def main():
    client = AsyncHIPAAQAClient()
    result = await client.ask_question("What is a covered entity?")
    print(result["answer"])

asyncio.run(main())
```

## 🔧 JavaScript/TypeScript Client

```typescript
interface QARequest {
  question: string;
  similarity_threshold?: number;
  max_chunks?: number;
  include_metadata?: boolean;
}

interface QAResponse {
  answer: string;
  sources: Array<{
    chunk_id: number;
    section: string;
    part: string;
    similarity_score: number;
    content: string;
  }>;
  metadata: {
    chunks_retrieved: number;
    processing_time_ms: number;
    model_used: string;
    confidence_score: number;
  };
}

class HIPAAQAClient {
  constructor(private baseUrl: string = 'http://localhost/api') {}

  async askQuestion(request: QARequest): Promise<QAResponse> {
    const response = await fetch(`${this.baseUrl}/qa/ask`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  }

  async healthCheck(): Promise<any> {
    const response = await fetch(`${this.baseUrl}/health/`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return response.json();
  }
}

// Usage
const client = new HIPAAQAClient();
const result = await client.askQuestion({
  question: "What is a covered entity?",
  similarity_threshold: 0.4
});
console.log(result.answer);
```

## 🧪 Testing

### Curl Examples

```bash
# Basic question
curl -X POST "http://localhost/api/qa/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is PHI?"}'

# With custom parameters
curl -X POST "http://localhost/api/qa/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the administrative safeguards?",
    "similarity_threshold": 0.3,
    "max_chunks": 10
  }' | jq '.answer'

# Health checks
curl "http://localhost/api/health/"
curl "http://localhost/api/health/ready"
curl "http://localhost/api/health/db"

# System info
curl "http://localhost/api/models"
curl "http://localhost/api/stats"
```

### Test Suite

```python
import pytest
import httpx

BASE_URL = "http://localhost/api"

@pytest.mark.asyncio
async def test_ask_question():
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{BASE_URL}/qa/ask",
            json={"question": "What is a business associate?"}
        )
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert len(data["sources"]) > 0

@pytest.mark.asyncio
async def test_health_endpoints():
    async with httpx.AsyncClient() as client:
        # Basic health
        response = await client.get(f"{BASE_URL}/health/")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"
        
        # Readiness check
        response = await client.get(f"{BASE_URL}/health/ready")
        assert response.status_code == 200
        data = response.json()
        assert data["database_connected"] is True
        assert data["chunks_indexed"] > 0

@pytest.mark.asyncio 
async def test_models_endpoint():
    async with httpx.AsyncClient() as client:
        response = await client.get(f"{BASE_URL}/models")
        assert response.status_code == 200
        data = response.json()
        assert "chat_model" in data
        assert "embedding_model" in data
```

## 🚨 Error Handling

### Error Response Format

```json
{
  "detail": "Error description",
  "type": "error_type",
  "code": "ERROR_CODE"
}
```

### Common HTTP Status Codes

- **200**: Success
- **400**: Bad Request (invalid parameters)
- **422**: Validation Error (malformed request)
- **429**: Rate Limit Exceeded
- **500**: Internal Server Error
- **503**: Service Unavailable (database/OpenAI issues)

### Error Examples

```bash
# Invalid question parameter
curl -X POST "http://localhost/api/qa/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": ""}'
# Response: 422 Validation Error

# Rate limit exceeded
curl -X POST "http://localhost/api/qa/ask" ... # (too many requests)
# Response: 429 Too Many Requests

# Service unavailable
curl -X POST "http://localhost/api/qa/ask" ... # (when database is down)
# Response: 503 Service Unavailable
```

## 📈 Performance Considerations

### Response Times

- **Simple questions**: 500-1500ms
- **Complex questions**: 1500-3000ms
- **Health checks**: <50ms

### Optimization Tips

1. **Adjust similarity threshold**: Lower values retrieve more chunks but take longer
2. **Limit max_chunks**: Fewer chunks = faster processing
3. **Use caching**: Implement client-side caching for repeated questions
4. **Batch requests**: If possible, batch multiple questions

### Monitoring

Monitor these metrics for optimal performance:

- Average response time
- Error rate
- Rate limit hits
- Database connection pool usage
- OpenAI API quota usage

---

For additional information, see the [main documentation](../README.md) or explore the interactive API docs at `/api/docs`.
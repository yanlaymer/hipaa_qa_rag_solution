# 🚀 Deployment Guide

This guide covers different deployment scenarios for the HIPAA QA System.

## 📋 Prerequisites

- Docker & Docker Compose
- OpenAI API key
- 8GB+ RAM (16GB recommended for production)
- 10GB+ disk space

## 🏠 Local Development

### 1. Quick Start

```bash
# Clone and setup
git clone https://github.com/your-username/hipaa-qa-system.git
cd hipaa-qa-system

# Configure environment
cp env.example .env
# Edit .env with your OpenAI API key

# Start all services
docker-compose up -d --build

# Check status
docker-compose ps
```

### 2. Development Mode

For active development with hot reloading:

```bash
# Start only database
docker-compose up -d db

# Install dependencies locally
pip install -e .

# Run backend locally (with auto-reload)
uvicorn src.hipaa_qa.main:app --reload --host 0.0.0.0 --port 8000

# Run frontend locally (in another terminal)
python frontend/app.py
```

## 🌐 Production Deployment

### Environment Configuration

Create production `.env`:

```env
# OpenAI Configuration
OPENAI_API_KEY=your_production_api_key
OPENAI_CHAT_MODEL=gpt-4-turbo
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# Database Configuration
POSTGRES_DB=hipaa_qa_prod
POSTGRES_USER=hipaa_qa_user
POSTGRES_PASSWORD=secure_random_password

# API Configuration
DEBUG=false
LOG_LEVEL=WARNING
LOG_FORMAT=json

# Performance Configuration
DB_POOL_SIZE=20
API_WORKERS=4
BATCH_SIZE=200

# Security
GRADIO_SHARE=false
```

### Docker Production

```bash
# Production deployment
docker-compose -f docker-compose.yml up -d --build

# Scale backend for load
docker-compose up -d --scale backend=3

# Monitor services
docker-compose logs -f
```

## ☁️ Cloud Deployment

### AWS Deployment

#### Option 1: ECS with Fargate

1. **Build and push images**:
```bash
# Build images
docker build -f Dockerfile.backend -t your-registry/hipaa-qa-backend .
docker build -f Dockerfile.frontend -t your-registry/hipaa-qa-frontend .
docker build -f Dockerfile.nginx -t your-registry/hipaa-qa-nginx .

# Push to ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin your-account.dkr.ecr.us-east-1.amazonaws.com
docker tag your-registry/hipaa-qa-backend:latest your-account.dkr.ecr.us-east-1.amazonaws.com/hipaa-qa-backend:latest
docker push your-account.dkr.ecr.us-east-1.amazonaws.com/hipaa-qa-backend:latest
```

2. **Create ECS task definition**:
```json
{
  "family": "hipaa-qa-system",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "backend",
      "image": "your-account.dkr.ecr.us-east-1.amazonaws.com/hipaa-qa-backend:latest",
      "portMappings": [{"containerPort": 8000}],
      "environment": [
        {"name": "OPENAI_API_KEY", "value": "your-key"},
        {"name": "DB_HOST", "value": "your-rds-endpoint"}
      ]
    }
  ]
}
```

#### Option 2: EC2 with Docker Compose

```bash
# On EC2 instance
sudo yum update -y
sudo yum install -y docker
sudo systemctl start docker
sudo systemctl enable docker

# Install docker-compose
sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Deploy application
git clone https://github.com/your-username/hipaa-qa-system.git
cd hipaa-qa-system
# Configure .env file
docker-compose up -d --build
```

### Google Cloud Platform

#### Cloud Run Deployment

1. **Prepare for Cloud Run**:
```dockerfile
# Dockerfile.cloudrun
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml ./
COPY src/ ./src/
COPY data/ ./data/

RUN pip install -e .

# Cloud Run expects service on $PORT
ENV PORT=8080
ENV API_HOST=0.0.0.0
ENV API_PORT=8080

CMD ["python", "-m", "src.hipaa_qa.main"]
```

2. **Deploy**:
```bash
# Build and deploy
gcloud builds submit --tag gcr.io/your-project/hipaa-qa-backend
gcloud run deploy hipaa-qa-backend \
  --image gcr.io/your-project/hipaa-qa-backend \
  --platform managed \
  --region us-central1 \
  --set-env-vars OPENAI_API_KEY=your-key \
  --memory 4Gi \
  --cpu 2 \
  --max-instances 10
```

### Azure Container Instances

```bash
# Create resource group
az group create --name hipaa-qa-rg --location eastus

# Deploy with container group
az container create \
  --resource-group hipaa-qa-rg \
  --name hipaa-qa-system \
  --image your-registry/hipaa-qa-backend:latest \
  --dns-name-label hipaa-qa-unique \
  --ports 8000 \
  --memory 4 \
  --cpu 2 \
  --environment-variables \
    OPENAI_API_KEY=your-key \
    DB_HOST=your-postgres-server
```

## 🔗 Load Balancing & Scaling

### Nginx Load Balancing

Update `nginx/nginx.conf` for multiple backend instances:

```nginx
upstream backend {
    server backend-1:8000;
    server backend-2:8000;
    server backend-3:8000;
    
    # Load balancing method
    least_conn;
}
```

### Docker Compose Scaling

```bash
# Scale backend instances
docker-compose up -d --scale backend=3

# Scale with different services
docker-compose up -d --scale backend=3 --scale frontend=2
```

### Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: hipaa-qa-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: hipaa-qa-backend
  template:
    metadata:
      labels:
        app: hipaa-qa-backend
    spec:
      containers:
      - name: backend
        image: your-registry/hipaa-qa-backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: openai-secret
              key: api-key
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
---
apiVersion: v1
kind: Service
metadata:
  name: hipaa-qa-backend-service
spec:
  selector:
    app: hipaa-qa-backend
  ports:
  - port: 8000
    targetPort: 8000
  type: LoadBalancer
```

## 🔒 Security Considerations

### Production Security Checklist

- [ ] Use strong, unique passwords for database
- [ ] Store API keys in secure vault (AWS Secrets Manager, etc.)
- [ ] Enable HTTPS/TLS encryption
- [ ] Configure firewall rules (allow only necessary ports)
- [ ] Set up log aggregation and monitoring
- [ ] Regular security updates
- [ ] Backup strategy implementation
- [ ] Rate limiting configuration
- [ ] CORS configuration for frontend

### Environment Variables Security

```bash
# Use external secret management
# AWS Secrets Manager
aws secretsmanager get-secret-value --secret-id openai-api-key

# HashiCorp Vault
vault kv get -field=api_key secret/openai

# Kubernetes Secrets
kubectl create secret generic openai-secret --from-literal=api-key=your-key
```

## 📊 Monitoring & Logging

### Production Monitoring Setup

1. **Log Aggregation**:
```bash
# ELK Stack
docker run -d -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" elasticsearch:7.14.0
docker run -d -p 5601:5601 --link elastic:elasticsearch kibana:7.14.0
```

2. **Metrics Collection**:
```bash
# Prometheus + Grafana
docker run -d -p 9090:9090 prom/prometheus
docker run -d -p 3000:3000 grafana/grafana
```

3. **Health Monitoring**:
```bash
# Set up health check endpoints
curl http://your-domain/health
curl http://your-domain/api/health/
```

### Application Performance Monitoring

```python
# Add APM instrumentation
pip install ddtrace  # Datadog
# or
pip install elastic-apm  # Elastic APM

# In your application
from ddtrace import tracer

@tracer.wrap("qa_processing")
def process_question(question):
    # Your code here
    pass
```

## 🔄 CI/CD Pipeline

### GitHub Actions

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Build Docker images
      run: |
        docker build -f Dockerfile.backend -t ${{ secrets.REGISTRY_URL }}/hipaa-qa-backend:${{ github.sha }} .
        docker build -f Dockerfile.frontend -t ${{ secrets.REGISTRY_URL }}/hipaa-qa-frontend:${{ github.sha }} .
    
    - name: Push to registry
      run: |
        echo ${{ secrets.REGISTRY_PASSWORD }} | docker login ${{ secrets.REGISTRY_URL }} -u ${{ secrets.REGISTRY_USERNAME }} --password-stdin
        docker push ${{ secrets.REGISTRY_URL }}/hipaa-qa-backend:${{ github.sha }}
        docker push ${{ secrets.REGISTRY_URL }}/hipaa-qa-frontend:${{ github.sha }}
    
    - name: Deploy to production
      run: |
        # Your deployment script here
        kubectl set image deployment/hipaa-qa-backend backend=${{ secrets.REGISTRY_URL }}/hipaa-qa-backend:${{ github.sha }}
```

## 📈 Performance Optimization

### Database Optimization

```sql
-- Create indexes for better performance
CREATE INDEX CONCURRENTLY idx_chunks_embedding_vector 
ON hipaa_chunks USING ivfflat (embedding_vector vector_cosine_ops);

-- Optimize PostgreSQL settings
-- postgresql.conf
shared_preload_libraries = 'pg_stat_statements,auto_explain'
max_connections = 200
shared_buffers = 256MB
effective_cache_size = 1GB
```

### Application Optimization

```python
# Use connection pooling
from sqlalchemy.pool import QueuePool

engine = create_async_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_pre_ping=True
)

# Implement caching
from functools import lru_cache

@lru_cache(maxsize=128)
def get_cached_embedding(text: str):
    return openai_client.embeddings.create(input=text)
```

## 🛟 Troubleshooting

### Common Issues

1. **Out of Memory**:
   - Increase Docker memory limits
   - Reduce batch sizes
   - Optimize database queries

2. **Slow Response Times**:
   - Check OpenAI API quotas
   - Optimize similarity thresholds
   - Add database indexes

3. **Connection Issues**:
   - Verify network configuration
   - Check firewall rules
   - Validate environment variables

### Debug Commands

```bash
# Check service health
docker-compose ps
docker-compose logs service-name

# Database connection test
docker-compose exec db psql -U postgres -d hipaa_qa -c "SELECT COUNT(*) FROM hipaa_chunks;"

# API endpoint test
curl -X POST http://localhost/api/qa/ask -H "Content-Type: application/json" -d '{"question": "test"}'
```

---

For additional support, check the [main README](../README.md) or open an issue on GitHub.
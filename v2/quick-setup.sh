#!/bin/bash
# quick_setup.sh - Quick setup for Requirements Agent

set -e

echo "🚀 Setting up Requirements Agent - Quick Deploy"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    log_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Create project structure
log_info "Creating project structure..."
mkdir -p logs data monitoring

# Create minimal requirements.txt for basic functionality
cat > requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
websockets==12.0
python-multipart==0.0.6
langchain==0.0.340
langchain-openai==0.0.2
openai==1.3.7
requests==2.31.0
httpx==0.25.2
aiofiles==23.2.1
pydantic==2.5.1
python-dotenv==1.0.0
PyPDF2==3.0.1
python-docx==0.8.11
mammoth==1.6.0
msal==1.25.0
atlassian-python-api==3.41.10
redis==5.0.1
neo4j==5.14.1
structlog==23.2.0
prometheus-client==0.19.0
EOF

# Create environment file with demo keys
cat > .env << 'EOF'
# Agentic AI SDLC - Requirements Agent Configuration

# Environment
ENVIRONMENT=development
PROJECT_NAME=requirements-agent
VERSION=1.0.0

# API Keys (Replace with your actual keys)
OPENAI_API_KEY=demo-key-replace-with-real
GEMINI_API_KEY=your-gemini-key-here
AZURE_OPENAI_ENDPOINT=your-azure-endpoint-here
AZURE_OPENAI_KEY=your-azure-key-here

# Database Credentials
REDIS_PASSWORD=agentredis
NEO4J_AUTH=neo4j/agentpassword

# Monitoring
GRAFANA_ADMIN_PASSWORD=agentgrafana

# External Tool Integrations (Optional)
JIRA_BASE_URL=https://your-company.atlassian.net
JIRA_EMAIL=your-email@company.com
JIRA_TOKEN=your-jira-token

CONFLUENCE_BASE_URL=https://your-company.atlassian.net/wiki
CONFLUENCE_TOKEN=your-confluence-token

# Logging
LOG_LEVEL=INFO
EOF

# Create Prometheus configuration
mkdir -p monitoring
cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'requirements-agent'
    static_configs:
      - targets: ['requirements-agent:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s
EOF

# Build and start the application
log_info "Building and starting Requirements Agent..."

# Stop any existing containers
docker-compose down -v 2>/dev/null || true

# Build and start services
docker-compose up -d --build

# Wait for services to be ready
log_info "Waiting for services to start..."
sleep 30

# Check health
log_info "Checking service health..."

# Check Requirements Agent
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    log_info "✅ Requirements Agent API is healthy"
else
    log_warning "⚠️  Requirements Agent API health check failed"
fi

# Check other services
if curl -f http://localhost:9090 > /dev/null 2>&1; then
    log_info "✅ Prometheus is accessible"
else
    log_warning "⚠️  Prometheus not accessible"
fi

if curl -f http://localhost:3001 > /dev/null 2>&1; then
    log_info "✅ Grafana is accessible"
else
    log_warning "⚠️  Grafana not accessible"
fi

echo ""
echo "🎉 Requirements Agent Setup Complete!"
echo ""
echo "📊 Access URLs:"
echo "  API Documentation:     http://localhost:8000/docs"
echo "  Health Check:          http://localhost:8000/health"
echo "  Grafana Monitoring:    http://localhost:3001 (admin/agentgrafana)"
echo "  Prometheus Metrics:    http://localhost:9090"
echo "  Neo4j Browser:         http://localhost:7474 (neo4j/agentpassword)"
echo ""
echo "🔧 Next Steps:"
echo "  1. Edit .env file with your actual API keys"
echo "  2. Test the API: curl http://localhost:8000/health"
echo "  3. Upload documents via API: curl -X POST -F 'files=@document.pdf' http://localhost:8000/api/upload"
echo "  4. View logs: docker-compose logs -f requirements-agent"
echo ""
echo "📚 API Examples:"
echo "  # Upload file"
echo "  curl -X POST -F 'files=@requirements.pdf' http://localhost:8000/api/upload"
echo ""
echo "  # Analyze requirements"
echo "  curl -X POST http://localhost:8000/api/agents/requirements/analyze \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"name\":\"Test Project\",\"description\":\"Sample project\"}'"
echo ""
echo "  # Connect to Confluence"
echo "  curl -X POST http://localhost:8000/api/integrations/connect \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"integration_type\":\"confluence\",\"config\":{\"baseUrl\":\"https://company.atlassian.net/wiki\",\"email\":\"user@company.com\",\"apiToken\":\"token\",\"spaceKey\":\"PROJ\"}}'"
echo ""

# Show running containers
echo "🐳 Running Containers:"
docker-compose ps

echo ""
log_info "Setup completed successfully!"
log_warning "Remember to update .env with your actual API keys!"

# Check if OPENAI_API_KEY is still demo
if grep -q "demo-key-replace-with-real" .env; then
    echo ""
    log_warning "🔑 IMPORTANT: Update OPENAI_API_KEY in .env file for full functionality!"
    echo "   Without a real OpenAI API key, the agent will use mock responses."
fi
#!/bin/bash
# deploy.sh - Deployment script for Agentic AI SDLC Requirements Agent

set -e

echo "🚀 Deploying Agentic AI SDLC - Requirements Agent"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="agentic-sdlc-requirements"
DOCKER_REGISTRY=${DOCKER_REGISTRY:-"your-registry.com"}
VERSION=${VERSION:-"1.0.0"}
ENVIRONMENT=${ENVIRONMENT:-"development"}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Create environment file
create_env_file() {
    log_info "Creating environment configuration..."
    
    if [ ! -f .env ]; then
        cat > .env << EOF
# Agentic AI SDLC - Requirements Agent Configuration

# Environment
ENVIRONMENT=${ENVIRONMENT}
PROJECT_NAME=${PROJECT_NAME}
VERSION=${VERSION}

# API Keys (Set your actual API keys)
OPENAI_API_KEY=your-openai-api-key-here
GEMINI_API_KEY=your-gemini-api-key-here
AZURE_OPENAI_ENDPOINT=your-azure-endpoint-here
AZURE_OPENAI_KEY=your-azure-key-here

# Database Credentials
REDIS_PASSWORD=agentredis
NEO4J_AUTH=neo4j/agentpassword

# Security
JWT_SECRET_KEY=your-jwt-secret-key-here
ENCRYPTION_KEY=your-encryption-key-here

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ADMIN_PASSWORD=agentgrafana
SENTRY_DSN=your-sentry-dsn-here

# External Tool Integrations
JIRA_BASE_URL=https://your-company.atlassian.net
JIRA_EMAIL=your-email@company.com
JIRA_TOKEN=your-jira-token

GITHUB_ORG=your-github-org
GITHUB_TOKEN=your-github-token

CONFLUENCE_BASE_URL=https://your-company.atlassian.net/wiki
CONFLUENCE_TOKEN=your-confluence-token

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json

# Resource Limits
MAX_CONCURRENT_AGENTS=10
AGENT_TIMEOUT_SECONDS=300
MEMORY_LIMIT=2g
CPU_LIMIT=1.0
EOF
        log_success "Environment file created. Please update with your actual API keys and configurations."
        log_warning "⚠️  Edit .env file with your actual API keys before deployment!"
    else
        log_info "Environment file already exists"
    fi
}

# Create monitoring configuration
create_monitoring_config() {
    log_info "Setting up monitoring configuration..."
    
    mkdir -p monitoring/prometheus monitoring/grafana/provisioning/dashboards monitoring/grafana/provisioning/datasources
    
    # Prometheus configuration
    cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'requirements-agent'
    static_configs:
      - targets: ['requirements-agent:8000']
    metrics_path: '/metrics'
    scrape_interval: 5s

  - job_name: 'redis'
    static_configs:
      - targets: ['redis:6379']

  - job_name: 'neo4j'
    static_configs:
      - targets: ['neo4j:7474']
EOF

    # Grafana datasource configuration
    cat > monitoring/grafana/provisioning/datasources/datasources.yml << 'EOF'
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
EOF

    # Basic Grafana dashboard
    cat > monitoring/grafana/provisioning/dashboards/dashboard.yml << 'EOF'
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOF

    log_success "Monitoring configuration created"
}

# Create nginx configuration
create_nginx_config() {
    log_info "Creating nginx configuration..."
    
    mkdir -p nginx
    
    cat > nginx/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream backend {
        server requirements-agent:8000;
    }

    upstream frontend {
        server frontend:3000;
    }

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=websocket:10m rate=5r/s;

    server {
        listen 80;
        server_name localhost;

        # Frontend
        location / {
            proxy_pass http://frontend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # API
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # WebSocket
        location /ws {
            limit_req zone=websocket burst=10 nodelay;
            proxy_pass http://backend;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Health check
        location /health {
            proxy_pass http://backend/health;
            access_log off;
        }
    }
}
EOF

    log_success "Nginx configuration created"
}

# Create ELK configuration
create_elk_config() {
    log_info "Creating ELK stack configuration..."
    
    mkdir -p elk/logstash/config elk/logstash/pipeline
    
    cat > elk/logstash/config/logstash.yml << 'EOF'
http.host: "0.0.0.0"
xpack.monitoring.elasticsearch.hosts: [ "http://elasticsearch:9200" ]
EOF

    cat > elk/logstash/pipeline/logstash.conf << 'EOF'
input {
  beats {
    port => 5044
  }
}

output {
  elasticsearch {
    hosts => "elasticsearch:9200"
    manage_template => false
    index => "%{[@metadata][beat]}-%{[@metadata][version]}-%{+YYYY.MM.dd}"
  }
}
EOF

    log_success "ELK configuration created"
}

# Build and deploy
build_and_deploy() {
    log_info "Building and deploying application..."
    
    # Pull latest images
    docker-compose pull
    
    # Build custom images
    docker-compose build --no-cache
    
    # Start services
    docker-compose up -d
    
    log_success "Application deployed successfully"
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Health check
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        log_success "Requirements Agent API is healthy"
    else
        log_error "Requirements Agent API health check failed"
    fi
    
    if curl -f http://localhost:3000 > /dev/null 2>&1; then
        log_success "Frontend is accessible"
    else
        log_error "Frontend health check failed"
    fi
}

# Display deployment information
show_deployment_info() {
    log_success "🎉 Deployment completed successfully!"
    echo
    echo "📊 Service URLs:"
    echo "  Frontend Dashboard:    http://localhost:3000"
    echo "  API Documentation:     http://localhost:8000/docs"
    echo "  Agent Health Check:    http://localhost:8000/health"
    echo "  Grafana Monitoring:    http://localhost:3001 (admin/agentgrafana)"
    echo "  Prometheus Metrics:    http://localhost:9090"
    echo "  Neo4j Browser:         http://localhost:7474 (neo4j/agentpassword)"
    echo "  Kibana Logs:           http://localhost:5601"
    echo
    echo "🔧 Management Commands:"
    echo "  View logs:             docker-compose logs -f"
    echo "  Stop services:         docker-compose down"
    echo "  Update services:       docker-compose pull && docker-compose up -d"
    echo "  Scale agents:          docker-compose up -d --scale requirements-agent=3"
    echo
    echo "📋 Next Steps:"
    echo "  1. Update .env file with your actual API keys"
    echo "  2. Configure external tool integrations (Jira, GitHub, etc.)"
    echo "  3. Set up SSL certificates for production"
    echo "  4. Configure monitoring alerts"
    echo "  5. Run test requirements gathering workflow"
    echo
}

# Cleanup function
cleanup() {
    log_info "Cleaning up resources..."
    docker-compose down -v
    docker system prune -f
    log_success "Cleanup completed"
}

# Main deployment flow
main() {
    case "${1:-deploy}" in
        "deploy")
            check_prerequisites
            create_env_file
            create_monitoring_config
            create_nginx_config
            create_elk_config
            build_and_deploy
            show_deployment_info
            ;;
        "cleanup")
            cleanup
            ;;
        "logs")
            docker-compose logs -f
            ;;
        "status")
            docker-compose ps
            ;;
        "restart")
            docker-compose restart
            ;;
        *)
            echo "Usage: $0 {deploy|cleanup|logs|status|restart}"
            echo "  deploy   - Full deployment (default)"
            echo "  cleanup  - Stop and remove all containers"
            echo "  logs     - Show application logs"
            echo "  status   - Show container status"
            echo "  restart  - Restart all services"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"
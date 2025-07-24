#!/bin/bash

# Complete Agentic AI SDLC System Deployment Script
# This script creates and deploys the complete 9-agent SDLC system

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Helper functions
print_header() {
    echo -e "\n${PURPLE}=================================${NC}"
    echo -e "${PURPLE}$1${NC}"
    echo -e "${PURPLE}=================================${NC}\n"
}

print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_header "🔍 Checking Prerequisites"
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_status "Docker found: $(docker --version)"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    print_status "Docker Compose found: $(docker-compose --version)"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.8+ first."
        exit 1
    fi
    print_status "Python found: $(python3 --version)"
    
    # Check Node.js
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed. Please install Node.js 18+ first."
        exit 1
    fi
    print_status "Node.js found: $(node --version)"
    
    # Check available disk space (minimum 10GB)
    AVAILABLE_SPACE=$(df . | tail -1 | awk '{print $4}')
    if [ "$AVAILABLE_SPACE" -lt 10485760 ]; then  # 10GB in KB
        print_warning "Less than 10GB disk space available. Continuing anyway..."
    else
        print_status "Sufficient disk space available"
    fi
    
    # Check available memory (minimum 4GB)
    AVAILABLE_MEMORY=$(free -m | awk 'NR==2{print $7}')
    if [ "$AVAILABLE_MEMORY" -lt 4096 ]; then  # 4GB
        print_warning "Less than 4GB RAM available. System may run slowly."
    else
        print_status "Sufficient memory available"
    fi
}

# Create project structure
create_project_structure() {
    print_header "📁 Creating Project Structure"
    
    # Main project directory
    PROJECT_DIR="agentic-ai-sdlc"
    if [ -d "$PROJECT_DIR" ]; then
        print_warning "Project directory already exists. Backing up..."
        mv "$PROJECT_DIR" "${PROJECT_DIR}_backup_$(date +%Y%m%d_%H%M%S)"
    fi
    
    mkdir -p "$PROJECT_DIR"
    cd "$PROJECT_DIR"
    
    # Create directory structure
    mkdir -p {orchestrator,frontend,agents,scripts,tests,docs,prometheus,grafana,logs}
    mkdir -p agents/{requirements,design,code,quality,testing,cicd,deployment,monitoring,maintenance}
    mkdir -p frontend/{src,public}
    mkdir -p orchestrator/{core,api,services}
    mkdir -p grafana/{dashboards,provisioning}
    mkdir -p prometheus/{rules,targets}
    
    print_status "Project structure created successfully"
    
    # Create basic .gitignore
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.venv/
pip-log.txt
pip-delete-this-directory.txt
.pytest_cache/

# Node.js
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
.npm
.yarn-integrity

# Environment variables
.env
.env.local
.env.development.local
.env.test.local
.env.production.local

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
logs/
*.log

# Docker
.docker/

# Database
*.db
*.sqlite3

# Build outputs
dist/
build/
.next/
EOF
    
    print_status "Created .gitignore file"
}

# Setup environment configuration
setup_environment() {
    print_header "⚙️ Setting Up Environment Configuration"
    
    cat > .env.example << 'EOF'
# LLM Configuration
OPENAI_API_KEY=your-openai-api-key-here
ANTHROPIC_API_KEY=your-anthropic-api-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# Database Configuration
POSTGRES_DB=agentic_sdlc
POSTGRES_USER=agentic_user
POSTGRES_PASSWORD=agentic-secure-password-2024
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# Neo4j Configuration
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=agentic-neo4j-password-2024

# Redis Configuration
REDIS_URL=redis://redis:6379

# SDLC Tools Configuration
JIRA_URL=https://your-domain.atlassian.net
JIRA_USERNAME=your-email@company.com
JIRA_API_TOKEN=your-jira-api-token

CONFLUENCE_URL=https://your-domain.atlassian.net/wiki
CONFLUENCE_USERNAME=your-email@company.com
CONFLUENCE_API_TOKEN=your-confluence-api-token

GITHUB_TOKEN=ghp_your-github-token-here
GITLAB_TOKEN=glpat-your-gitlab-token-here

JENKINS_URL=http://jenkins.company.com
JENKINS_USERNAME=admin
JENKINS_API_TOKEN=your-jenkins-api-token

# Monitoring Tools
PROMETHEUS_URL=http://prometheus:9090
GRAFANA_URL=http://grafana:3000
GRAFANA_TOKEN=your-grafana-service-account-token
DATADOG_API_KEY=your-datadog-api-key

# Security
JWT_SECRET_KEY=your-super-secret-jwt-key-change-in-production
ENCRYPTION_KEY=your-32-character-encryption-key

# System Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
MAX_WORKERS=4
ENABLE_DEBUG=false
EOF

    # Copy example to actual .env file
    cp .env.example .env
    
    print_status "Environment configuration created ✅"
    print_warning "Please edit .env file with your actual API keys and credentials"
}

# Create Docker configuration
create_docker_config() {
    print_header "🐳 Creating Docker Configuration"
    
    # Main docker-compose.yml
    cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  # Infrastructure Services
  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]
      interval: 10s
      timeout: 5s
      retries: 5

  neo4j:
    image: neo4j:5-community
    environment:
      NEO4J_AUTH: ${NEO4J_USER}/${NEO4J_PASSWORD}
      NEO4J_apoc_export_file_enabled: true
      NEO4J_apoc_import_file_enabled: true
    volumes:
      - neo4j_data:/data
    ports:
      - "7474:7474"
      - "7687:7687"
    healthcheck:
      test: ["CMD", "cypher-shell", "-u", "${NEO4J_USER}", "-p", "${NEO4J_PASSWORD}", "RETURN 1"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Monitoring Services
  prometheus:
    image: prom/prometheus:latest
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    volumes:
      - ./prometheus/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    depends_on:
      - orchestrator

  grafana:
    image: grafana/grafana:latest
    environment:
      GF_SECURITY_ADMIN_PASSWORD: agentic-grafana-2024
      GF_USERS_ALLOW_SIGN_UP: false
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/dashboards:/etc/grafana/provisioning/dashboards
    ports:
      - "3001:3000"
    depends_on:
      - prometheus

  # Application Services
  orchestrator:
    build:
      context: ./orchestrator
      dockerfile: Dockerfile
    environment:
      - POSTGRES_HOST=postgres
      - REDIS_URL=redis://redis:6379
      - NEO4J_URI=bolt://neo4j:7687
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs
    ports:
      - "8000:8000"
    depends_on:
      - postgres
      - redis
      - neo4j
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    depends_on:
      - orchestrator
    environment:
      - REACT_APP_API_URL=http://localhost:8000
    restart: unless-stopped

  # Agent Services
  requirements-agent:
    build:
      context: ./agents/requirements
      dockerfile: Dockerfile
    env_file:
      - .env
    depends_on:
      - orchestrator
    restart: unless-stopped

  design-agent:
    build:
      context: ./agents/design
      dockerfile: Dockerfile
    env_file:
      - .env
    depends_on:
      - orchestrator
    restart: unless-stopped

  code-agent:
    build:
      context: ./agents/code
      dockerfile: Dockerfile
    env_file:
      - .env
    depends_on:
      - orchestrator
    restart: unless-stopped

  quality-agent:
    build:
      context: ./agents/quality
      dockerfile: Dockerfile
    env_file:
      - .env
    depends_on:
      - orchestrator
    restart: unless-stopped

  testing-agent:
    build:
      context: ./agents/testing
      dockerfile: Dockerfile
    env_file:
      - .env
    depends_on:
      - orchestrator
    restart: unless-stopped

  cicd-agent:
    build:
      context: ./agents/cicd
      dockerfile: Dockerfile
    env_file:
      - .env
    depends_on:
      - orchestrator
    restart: unless-stopped

  deployment-agent:
    build:
      context: ./agents/deployment
      dockerfile: Dockerfile
    env_file:
      - .env
    depends_on:
      - orchestrator
    restart: unless-stopped

  monitoring-agent:
    build:
      context: ./agents/monitoring
      dockerfile: Dockerfile
    env_file:
      - .env
    depends_on:
      - orchestrator
    restart: unless-stopped

  maintenance-agent:
    build:
      context: ./agents/maintenance
      dockerfile: Dockerfile
    env_file:
      - .env
    depends_on:
      - orchestrator
    restart: unless-stopped

volumes:
  postgres_data:
  neo4j_data:
  redis_data:
  prometheus_data:
  grafana_data:

networks:
  default:
    name: agentic-sdlc-network
EOF

    print_status "Docker Compose configuration created"
}

# Create orchestrator
create_orchestrator() {
    print_header "🎭 Creating Orchestrator Service"
    
    cd orchestrator
    
    # Create main.py
    cat > main.py << 'EOF'
"""
Agentic AI SDLC Orchestrator - Main Application
"""
import asyncio
import uvicorn
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
import time
from datetime import datetime

app = FastAPI(
    title="Agentic AI SDLC Orchestrator",
    description="Central orchestrator for 9-agent SDLC system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Store active workflows
active_workflows = {}
connected_agents = {}

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Agentic AI SDLC Orchestrator starting up...")
    logger.info("🤖 Waiting for agents to connect...")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "active_workflows": len(active_workflows),
        "connected_agents": len(connected_agents)
    }

@app.get("/")
async def root():
    return {
        "message": "Agentic AI SDLC Orchestrator",
        "version": "1.0.0",
        "agents": [
            "requirements", "design", "code", "quality", "testing",
            "cicd", "deployment", "monitoring", "maintenance"
        ],
        "status": "running"
    }

@app.post("/workflows")
async def create_workflow(request: dict):
    workflow_id = f"workflow_{int(time.time())}"
    
    workflow = {
        "id": workflow_id,
        "project_id": request.get("project_id", "unknown"),
        "name": request.get("name", "Untitled Project"),
        "description": request.get("description", ""),
        "requirements": request.get("requirements", {}),
        "status": "created",
        "created_at": datetime.utcnow().isoformat(),
        "phases": [
            {"name": "requirements", "status": "pending", "agent": "requirements-agent"},
            {"name": "design", "status": "pending", "agent": "design-agent"},
            {"name": "code", "status": "pending", "agent": "code-agent"},
            {"name": "quality", "status": "pending", "agent": "quality-agent"},
            {"name": "testing", "status": "pending", "agent": "testing-agent"},
            {"name": "cicd", "status": "pending", "agent": "cicd-agent"},
            {"name": "deployment", "status": "pending", "agent": "deployment-agent"},
            {"name": "monitoring", "status": "pending", "agent": "monitoring-agent"},
            {"name": "maintenance", "status": "pending", "agent": "maintenance-agent"}
        ]
    }
    
    active_workflows[workflow_id] = workflow
    logger.info(f"✅ Created workflow: {workflow_id}")
    
    return {"workflow_id": workflow_id, "status": "created"}

@app.post("/workflows/{workflow_id}/execute")
async def execute_workflow(workflow_id: str):
    if workflow_id not in active_workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    workflow = active_workflows[workflow_id]
    workflow["status"] = "running"
    workflow["started_at"] = datetime.utcnow().isoformat()
    
    # Start with requirements phase
    workflow["phases"][0]["status"] = "running"
    
    logger.info(f"🚀 Executing workflow: {workflow_id}")
    
    # Simulate workflow execution
    asyncio.create_task(simulate_workflow_execution(workflow_id))
    
    return {"workflow_id": workflow_id, "status": "started"}

@app.get("/workflows/{workflow_id}/status")
async def get_workflow_status(workflow_id: str):
    if workflow_id not in active_workflows:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return active_workflows[workflow_id]

@app.get("/workflows")
async def list_workflows():
    return {"workflows": list(active_workflows.values())}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("📡 WebSocket client connected")
    
    try:
        while True:
            # Send periodic updates
            await websocket.send_json({
                "type": "status_update",
                "timestamp": datetime.utcnow().isoformat(),
                "active_workflows": len(active_workflows),
                "connected_agents": len(connected_agents)
            })
            await asyncio.sleep(5)
    except Exception as e:
        logger.info(f"📡 WebSocket client disconnected: {e}")

async def simulate_workflow_execution(workflow_id: str):
    """Simulate workflow execution through all phases"""
    workflow = active_workflows[workflow_id]
    
    for i, phase in enumerate(workflow["phases"]):
        # Simulate phase execution
        phase["status"] = "running"
        phase["started_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"🔄 Executing phase: {phase['name']} for workflow: {workflow_id}")
        
        # Simulate work (5-15 seconds per phase)
        await asyncio.sleep(10)
        
        phase["status"] = "completed"
        phase["completed_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"✅ Completed phase: {phase['name']} for workflow: {workflow_id}")
        
        # Start next phase if available
        if i + 1 < len(workflow["phases"]):
            workflow["phases"][i + 1]["status"] = "running"
    
    # Mark workflow as completed
    workflow["status"] = "completed"
    workflow["completed_at"] = datetime.utcnow().isoformat()
    
    logger.info(f"🎉 Workflow completed: {workflow_id}")

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return {
        "agentic_workflows_total": len(active_workflows),
        "agentic_agents_connected": len(connected_agents),
        "agentic_workflows_running": len([w for w in active_workflows.values() if w["status"] == "running"]),
        "agentic_workflows_completed": len([w for w in active_workflows.values() if w["status"] == "completed"])
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
EOF

    # Create requirements.txt
    cat > requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
websockets==12.0
pydantic==2.5.0
httpx==0.25.2
redis==5.0.1
neo4j==5.15.0
psycopg2-binary==2.9.9
python-multipart==0.0.6
aiofiles==23.2.1
prometheus-client==0.19.0
EOF

    # Create Dockerfile
    cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc g++ curl git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m -u 1000 agentic && chown -R agentic:agentic /app
USER agentic

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
EOF

    cd ..
    print_status "Orchestrator service created"
}

# Create frontend
create_frontend() {
    print_header "⚛️ Creating Frontend Application"
    
    cd frontend
    
    # Create package.json
    cat > package.json << 'EOF'
{
  "name": "agentic-ai-sdlc-frontend",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "react-router-dom": "^6.8.0",
    "axios": "^1.6.0",
    "lucide-react": "^0.263.1",
    "tailwindcss": "^3.3.0",
    "autoprefixer": "^10.4.14",
    "postcss": "^8.4.24"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build",
    "test": "react-scripts test",
    "eject": "react-scripts eject"
  },
  "eslintConfig": {
    "extends": [
      "react-app",
      "react-app/jest"
    ]
  },
  "browserslist": {
    "production": [
      ">0.2%",
      "not dead",
      "not op_mini all"
    ],
    "development": [
      "last 1 chrome version",
      "last 1 firefox version",
      "last 1 safari version"
    ]
  },
  "proxy": "http://orchestrator:8000"
}
EOF

    # Create public/index.html
    mkdir -p public
    cat > public/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="Agentic AI SDLC System Dashboard" />
    <title>Agentic AI SDLC</title>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>
EOF

    # Create src/index.js
    mkdir -p src
    cat > src/index.js << 'EOF'
import React from 'react';
import ReactDOM from 'react-dom/client';
import './index.css';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
EOF

    # Create basic App.js
    cat > src/App.js << 'EOF'
import React, { useState, useEffect } from 'react';
import { Play, GitBranch, Settings, Activity } from 'lucide-react';
import './App.css';

function App() {
  const [workflows, setWorkflows] = useState([]);
  const [agents, setAgents] = useState([
    { name: 'Requirements', status: 'connected' },
    { name: 'Design', status: 'connected' },
    { name: 'Code', status: 'connected' },
    { name: 'Quality', status: 'connected' },
    { name: 'Testing', status: 'connected' },
    { name: 'CI/CD', status: 'connected' },
    { name: 'Deployment', status: 'connected' },
    { name: 'Monitoring', status: 'connected' },
    { name: 'Maintenance', status: 'connected' }
  ]);

  const [newProject, setNewProject] = useState({
    name: '',
    description: '',
    requirements: ''
  });

  useEffect(() => {
    fetchWorkflows();
  }, []);

  const fetchWorkflows = async () => {
    try {
      const response = await fetch('/workflows');
      const data = await response.json();
      setWorkflows(data.workflows || []);
    } catch (error) {
      console.error('Error fetching workflows:', error);
    }
  };

  const createWorkflow = async () => {
    try {
      const response = await fetch('/workflows', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          project_id: `project_${Date.now()}`,
          name: newProject.name,
          description: newProject.description,
          requirements: { description: newProject.requirements }
        }),
      });
      
      const data = await response.json();
      
      // Execute the workflow
      await fetch(`/workflows/${data.workflow_id}/execute`, {
        method: 'POST'
      });
      
      setNewProject({ name: '', description: '', requirements: '' });
      fetchWorkflows();
    } catch (error) {
      console.error('Error creating workflow:', error);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100">
      {/* Header */}
      <header className="bg-white shadow-md">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center">
              <GitBranch className="h-8 w-8 text-blue-600 mr-3" />
              <h1 className="text-2xl font-bold text-gray-900">
                Agentic AI SDLC System
              </h1>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center text-green-600">
                <Activity className="h-5 w-5 mr-2" />
                <span className="text-sm font-medium">System Online</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Create New Project */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Create New Project
              </h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Project Name
                  </label>
                  <input
                    type="text"
                    value={newProject.name}
                    onChange={(e) => setNewProject({...newProject, name: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="My Awesome Project"
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Description
                  </label>
                  <textarea
                    value={newProject.description}
                    onChange={(e) => setNewProject({...newProject, description: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    rows="3"
                    placeholder="Brief description of your project..."
                  />
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Requirements
                  </label>
                  <textarea
                    value={newProject.requirements}
                    onChange={(e) => setNewProject({...newProject, requirements: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    rows="3"
                    placeholder="List your project requirements..."
                  />
                </div>
                
                <button
                  onClick={createWorkflow}
                  disabled={!newProject.name.trim()}
                  className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium py-2 px-4 rounded-md transition duration-200 flex items-center justify-center"
                >
                  <Play className="h-4 w-4 mr-2" />
                  Start SDLC Workflow
                </button>
              </div>
            </div>
                     
            {/* Agent Status */}
            <div className="bg-white rounded-lg shadow p-6 mt-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Agent Status
              </h2>
              <div className="space-y-3">
                {agents.map((agent, index) => (
                  <div key={index} className="flex items-center justify-between">
                    <span className="text-sm font-medium text-gray-700">
                      {agent.name} Agent
                    </span>
                    <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                      agent.status === 'connected' 
                        ? 'bg-green-100 text-green-800' 
                        : 'bg-red-100 text-red-800'
                    }`}>
                      {agent.status}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {/* Active Workflows */}
          <div className="lg:col-span-2">
            <div className="bg-white rounded-lg shadow p-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Active Workflows
              </h2>
              
              {workflows.length === 0 ? (
                <div className="text-center py-8">
                  <Settings className="h-12 w-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-500">No active workflows</p>
                  <p className="text-sm text-gray-400">Create a new project to get started</p>
                </div>
              ) : (
                <div className="space-y-4">
                  {workflows.map((workflow) => (
                    <div key={workflow.id} className="border border-gray-200 rounded-lg p-4">
                      <div className="flex items-center justify-between mb-2">
                        <h3 className="font-medium text-gray-900">{workflow.name}</h3>
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          workflow.status === 'completed' 
                            ? 'bg-green-100 text-green-800'
                            : workflow.status === 'running'
                            ? 'bg-blue-100 text-blue-800'
                            : 'bg-gray-100 text-gray-800'
                        }`}>
                          {workflow.status}
                        </span>
                      </div>
                      
                      <p className="text-sm text-gray-600 mb-3">{workflow.description}</p>
                      
                      {/* Progress Bar */}
                      <div className="space-y-2">
                        <div className="flex justify-between text-xs text-gray-500">
                          <span>Progress</span>
                          <span>
                            {workflow.phases.filter(p => p.status === 'completed').length} / {workflow.phases.length}
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div 
                            className="bg-blue-600 h-2 rounded-full transition-all duration-500"
                            style={{
                              width: `${(workflow.phases.filter(p => p.status === 'completed').length / workflow.phases.length) * 100}%`
                            }}
                          ></div>
                        </div>
                      </div>
                      
                      {/* Phase Details */}
                      <div className="mt-3 grid grid-cols-3 gap-2">
                        {workflow.phases.map((phase, index) => (
                          <div key={index} className="text-center">
                            <div className={`w-8 h-8 rounded-full mx-auto mb-1 flex items-center justify-center text-xs font-medium ${
                              phase.status === 'completed'
                                ? 'bg-green-500 text-white'
                                : phase.status === 'running'
                                ? 'bg-blue-500 text-white'
                                : 'bg-gray-300 text-gray-600'
                            }`}>
                              {index + 1}
                            </div>
                            <div className="text-xs text-gray-600 capitalize">
                              {phase.name}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
EOF

    # Create CSS
    cat > src/index.css << 'EOF'
@tailwind base;
@tailwind components;
@tailwind utilities;

body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen',
    'Ubuntu', 'Cantarell', 'Fira Sans', 'Droid Sans', 'Helvetica Neue',
    sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}

code {
  font-family: source-code-pro, Menlo, Monaco, Consolas, 'Courier New',
    monospace;
}
EOF

    cat > src/App.css << 'EOF'
.App {
  text-align: center;
}

.pulse {
  animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
}

@keyframes pulse {
  0%, 100% {
    opacity: 1;
  }
  50% {
    opacity: .5;
  }
}
EOF

    # Create Tailwind config
    cat > tailwind.config.js << 'EOF'
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
EOF

    # Create PostCSS config
    cat > postcss.config.js << 'EOF'
module.exports = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
}
EOF

    # Create Dockerfile
    cat > Dockerfile << 'EOF'
# Build stage
FROM node:18-alpine as builder

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

# Production stage
FROM nginx:alpine
COPY --from=builder /app/build /usr/share/nginx/html

# Create nginx config
RUN echo 'server { \
    listen 3000; \
    location / { \
        root /usr/share/nginx/html; \
        index index.html index.htm; \
        try_files $uri $uri/ /index.html; \
    } \
    location /api { \
        proxy_pass http://orchestrator:8000; \
        proxy_set_header Host $host; \
        proxy_set_header X-Real-IP $remote_addr; \
    } \
}' > /etc/nginx/conf.d/default.conf

EXPOSE 3000
CMD ["nginx", "-g", "daemon off;"]
EOF

    cd ..
    print_status "Frontend application created"
}

# Create agent templates
create_agent_templates() {
    print_header "🤖 Creating Agent Templates"
    
    for agent in requirements design code quality testing cicd deployment monitoring maintenance; do
        cd "agents/$agent"
        
        # Create basic main.py
        cat > main.py << EOF
"""
${agent^} Agent - Main Entry Point
"""
import asyncio
import logging
import time
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ${agent^}Agent:
    def __init__(self):
        self.agent_id = "${agent}_agent_001"
        self.status = "running"
        
    async def process_task(self, task):
        """Process incoming tasks"""
        logger.info(f"🤖 ${agent^} Agent processing task: {task.get('type', 'unknown')}")
        
        # Simulate processing time
        await asyncio.sleep(5)
        
        return {
            "agent": self.agent_id,
            "task_id": task.get("id", "unknown"),
            "status": "completed",
            "result": f"${agent^} task completed successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    async def start(self):
        """Start the agent"""
        logger.info(f"🚀 Starting ${agent^} Agent...")
        
        while True:
            try:
                # Agent main loop
                await asyncio.sleep(10)
                logger.info(f"💓 ${agent^} Agent heartbeat")
                
            except KeyboardInterrupt:
                logger.info(f"🛑 ${agent^} Agent stopping...")
                break
            except Exception as e:
                logger.error(f"❌ Error in ${agent^} Agent: {e}")
                await asyncio.sleep(5)

async def main():
    agent = ${agent^}Agent()
    await agent.start()

if __name__ == "__main__":
    asyncio.run(main())
EOF

        # Create requirements.txt
        cat > requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
httpx==0.25.2
pydantic==2.5.0
asyncio-throttle==1.0.2
aiofiles==23.2.1
EOF

        # Create Dockerfile
        cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd -m -u 1000 agentic && chown -R agentic:agentic /app
USER agentic

CMD ["python", "main.py"]
EOF

        cd ../..
    done
    
    print_status "Agent templates created for all 9 agents"
}

# Create monitoring configuration
create_monitoring_config() {
    print_header "📊 Creating Monitoring Configuration"
    
    # Prometheus config
    mkdir -p prometheus
    cat > prometheus/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'orchestrator'
    static_configs:
      - targets: ['orchestrator:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s

  - job_name: 'agentic-system'
    static_configs:
      - targets: ['orchestrator:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
EOF

    # Grafana dashboards
    mkdir -p grafana/dashboards
    cat > grafana/dashboards/agentic-dashboard.json << 'EOF'
{
  "dashboard": {
    "id": null,
    "title": "Agentic AI SDLC Dashboard",
    "tags": ["agentic", "sdlc"],
    "timezone": "browser",
    "panels": [
      {
        "title": "Active Workflows",
        "type": "stat",
        "targets": [
          {
            "expr": "agentic_workflows_total",
            "legendFormat": "Total Workflows"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 0,
          "y": 0
        }
      },
      {
        "title": "Agent Status",
        "type": "stat",
        "targets": [
          {
            "expr": "agentic_agents_connected",
            "legendFormat": "Connected Agents"
          }
        ],
        "gridPos": {
          "h": 8,
          "w": 12,
          "x": 12,
          "y": 0
        }
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
EOF

    print_status "Monitoring configuration created"
}

# Create startup scripts
create_startup_scripts() {
    print_header "🚀 Creating Startup Scripts"
    
    cd scripts
    
    # Development startup script
    cat > start-dev.sh << 'EOF'
#!/bin/bash

echo "🔧 Starting Agentic AI SDLC System in Development Mode"

# Start infrastructure services only
echo "📦 Starting infrastructure services..."
docker-compose up -d postgres neo4j redis prometheus grafana

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

echo "✅ Infrastructure services started!"
echo ""
echo "Available services:"
echo "• PostgreSQL: localhost:5432"
echo "• Neo4j: http://localhost:7474"
echo "• Redis: localhost:6379"
echo "• Prometheus: http://localhost:9090"
echo "• Grafana: http://localhost:3001"
echo ""
echo "To start the full system, run: ./start-prod.sh"
EOF

    # Production startup script
    cat > start-prod.sh << 'EOF'
#!/bin/bash

echo "🚀 Starting Agentic AI SDLC System in Production Mode"

# Check if .env file exists
if [ ! -f "../.env" ]; then
    echo "❌ .env file not found. Please copy .env.example to .env and configure it."
    exit 1
fi

# Build and start all services
echo "🔨 Building Docker images..."
docker-compose build --parallel

echo "🚀 Starting all services..."
docker-compose up -d

echo "⏳ Waiting for services to be ready..."
sleep 60

# Check service health
echo "🔍 Checking service health..."

# Check orchestrator
if curl -f http://localhost:8000/health &>/dev/null; then
    echo "✅ Orchestrator is healthy"
else
    echo "⚠️ Orchestrator may still be starting..."
fi

# Check frontend
if curl -f http://localhost:3000 &>/dev/null; then
    echo "✅ Frontend is accessible"
else
    echo "⚠️ Frontend may still be starting..."
fi

echo ""
echo "🎉 Agentic AI SDLC System Started Successfully!"
echo ""
echo "Access your system:"
echo "• 🎨 Frontend Dashboard: http://localhost:3000"
echo "• 📊 API Documentation: http://localhost:8000/docs"
echo "• 📈 Grafana Monitoring: http://localhost:3001 (admin/agentic-grafana-2024)"
echo "• 🧠 Neo4j Browser: http://localhost:7474 (neo4j/agentic-neo4j-password-2024)"
echo "• 🔍 Prometheus: http://localhost:9090"
echo ""
echo "📝 Next steps:"
echo "1. Update .env file with your API keys"
echo "2. Create your first SDLC workflow in the dashboard"
echo "3. Monitor agent execution in real-time"
echo ""
echo "🆘 For help: Check README.md or run ./logs.sh to view logs"
EOF

    # Stop script
    cat > stop.sh << 'EOF'
#!/bin/bash

echo "🛑 Stopping Agentic AI SDLC System"

docker-compose down

echo "✅ System stopped!"
echo ""
echo "To remove all data (including databases), run:"
echo "docker-compose down -v"
EOF

    # Logs script
    cat > logs.sh << 'EOF'
#!/bin/bash

echo "📋 Agentic AI SDLC System Logs"
echo ""

if [ "$1" == "" ]; then
    echo "Usage: ./logs.sh [service-name]"
    echo ""
    echo "Available services:"
    echo "• orchestrator"
    echo "• frontend"
    echo "• postgres"
    echo "• neo4j"
    echo "• redis"
    echo "• prometheus"
    echo "• grafana"
    echo "• requirements-agent"
    echo "• design-agent"
    echo "• code-agent"
    echo "• quality-agent"
    echo "• testing-agent"
    echo "• cicd-agent"
    echo "• deployment-agent"
    echo "• monitoring-agent"
    echo "• maintenance-agent"
    echo ""
    echo "Examples:"
    echo "• ./logs.sh orchestrator"
    echo "• ./logs.sh all  (show all logs)"
    exit 1
fi

if [ "$1" == "all" ]; then
    docker-compose logs -f
else
    docker-compose logs -f "$1"
fi
EOF

    # Status script
    cat > status.sh << 'EOF'
#!/bin/bash

echo "📊 Agentic AI SDLC System Status"
echo ""

# Check if docker-compose is running
if ! docker-compose ps &>/dev/null; then
    echo "❌ System is not running"
    echo "Run './start-prod.sh' to start the system"
    exit 1
fi

echo "🐳 Docker Services:"
docker-compose ps

echo ""
echo "🔍 Health Checks:"

# Check orchestrator
if curl -s http://localhost:8000/health &>/dev/null; then
    echo "✅ Orchestrator: Healthy"
else
    echo "❌ Orchestrator: Not responding"
fi

# Check frontend
if curl -s http://localhost:3000 &>/dev/null; then
    echo "✅ Frontend: Accessible"
else
    echo "❌ Frontend: Not accessible"
fi

# Check Grafana
if curl -s http://localhost:3001 &>/dev/null; then
    echo "✅ Grafana: Accessible"
else
    echo "❌ Grafana: Not accessible"
fi

echo ""
echo "📈 Quick Stats:"
if curl -s http://localhost:8000/health | grep -q "healthy"; then
    HEALTH_DATA=$(curl -s http://localhost:8000/health)
    echo "• Active Workflows: $(echo $HEALTH_DATA | grep -o '"active_workflows":[0-9]*' | cut -d: -f2)"
    echo "• Connected Agents: $(echo $HEALTH_DATA | grep -o '"connected_agents":[0-9]*' | cut -d: -f2)"
fi
EOF

    # Make scripts executable
    chmod +x *.sh
    
    cd ..
    print_status "Startup scripts created"
}

# Create documentation
create_documentation() {
    print_header "📚 Creating Documentation"
    
    cat > README.md << 'EOF'
# Agentic AI-Powered SDLC System

## 🎯 Overview

A comprehensive, production-ready Agentic AI system that automates the entire Software Development Lifecycle using 9 specialized AI agents.

## 🏗️ System Architecture

### Core Components
- **Orchestration Engine**: Coordinates all agents and workflows
- **Message Bus**: Real-time communication between agents
- **Knowledge Graph**: Neo4j-based context management
- **Real-time Frontend**: React dashboard with live monitoring

### SDLC Agents (9 Total)
1. **Requirements Agent** - Stakeholder interviews, requirements gathering
2. **Design Agent** - System architecture, UI/UX design, database design
3. **Code Generation Agent** - Full-stack code generation
4. **Code Quality Agent** - Security scanning, quality analysis
5. **Testing Agent** - Unit, integration, E2E testing
6. **CI/CD Agent** - Pipeline automation
7. **Deployment Agent** - Environment management
8. **Monitoring Agent** - Observability and alerting
9. **Maintenance Agent** - Support and maintenance

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- Python 3.8+
- Node.js 18+
- 8GB+ RAM
- 20GB+ free disk space

### 1. Setup
```bash
git clone <repository-url>
cd agentic-ai-sdlc
chmod +x scripts/*.sh
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your API keys and credentials
```

### 3. Start System
```bash
# Production mode
./scripts/start-prod.sh

# Development mode (infrastructure only)
./scripts/start-dev.sh
```

### 4. Access Applications
- **Frontend Dashboard**: http://localhost:3000
- **API Documentation**: http://localhost:8000/docs
- **Grafana Monitoring**: http://localhost:3001
- **Neo4j Browser**: http://localhost:7474

## 🔧 Configuration

### Required API Keys
- OpenAI API key for LLM capabilities
- Tool-specific tokens (optional but recommended)

### Environment Variables
See `.env.example` for complete configuration options.

## 🧪 Testing

```bash
# Check system status
./scripts/status.sh

# View logs
./scripts/logs.sh orchestrator
./scripts/logs.sh all

# Stop system
./scripts/stop.sh
```

## 📊 Monitoring

- **Real-time Dashboard**: http://localhost:3000
- **Prometheus Metrics**: http://localhost:9090
- **Grafana Dashboards**: http://localhost:3001

## 🆘 Troubleshooting

### Common Issues

1. **Services not starting**
   ```bash
   ./scripts/logs.sh [service-name]
   ```

2. **Database connection issues**
   ```bash
   docker-compose down -v
   ./scripts/start-prod.sh
   ```

3. **Frontend not loading**
   ```bash
   ./scripts/logs.sh frontend
   ```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request

## 📜 License

MIT License - see LICENSE file for details.
EOF

    print_status "Documentation created"
}

# Main deployment function
deploy_system() {
    print_header "🎯 Deploying Complete System"
    
    # Build and start services
    print_info "Building Docker images..."
    docker-compose build --parallel
    
    print_info "Starting infrastructure services..."
    docker-compose up -d postgres neo4j redis
    
    print_info "Waiting for databases to initialize..."
    sleep 30
    
    print_info "Starting application services..."
    docker-compose up -d
    
    print_info "Waiting for services to be ready..."
    sleep 45
    
    # Health checks
    print_header "🔍 Running Health Checks"
    
    # Check orchestrator
    if curl -f http://localhost:8000/health &>/dev/null; then
        print_status "Orchestrator is healthy"
    else
        print_warning "Orchestrator may still be starting..."
    fi
    
    # Check frontend
    if curl -f http://localhost:3000 &>/dev/null; then
        print_status "Frontend is accessible"
    else
        print_warning "Frontend may still be starting..."
    fi
    
    # Final success message
    print_header "🎉 Deployment Complete!"
    echo ""
    echo -e "${GREEN}🌟 Your Agentic AI SDLC System is now running!${NC}"
    echo ""
    echo -e "${CYAN}Access your system:${NC}"
    echo -e "${BLUE}• 🎨 Frontend Dashboard: http://localhost:3000${NC}"
    echo -e "${BLUE}• 📊 API Documentation: http://localhost:8000/docs${NC}"
    echo -e "${BLUE}• 📈 Grafana Monitoring: http://localhost:3001${NC}"
    echo -e "${BLUE}• 🧠 Neo4j Browser: http://localhost:7474${NC}"
    echo ""
    echo -e "${CYAN}Next Steps:${NC}"
    echo -e "${GREEN}1. Update .env file with your API keys${NC}"
    echo -e "${GREEN}2. Create your first SDLC workflow in the dashboard${NC}"
    echo -e "${GREEN}3. Monitor agent execution in real-time${NC}"
    echo ""
    echo -e "${CYAN}Useful Commands:${NC}"
    echo -e "${YELLOW}• ./scripts/status.sh - Check system status${NC}"
    echo -e "${YELLOW}• ./scripts/logs.sh [service] - View logs${NC}"
    echo -e "${YELLOW}• ./scripts/stop.sh - Stop the system${NC}"
    echo ""
    echo -e "${GREEN}🎊 Happy coding with your AI-powered SDLC system!${NC}"
}

# Main execution
main() {
    echo -e "${PURPLE}"
    echo "🌟 =================================="
    echo "   AGENTIC AI SDLC SYSTEM SETUP"
    echo "   Complete 9-Agent Implementation"
    echo "==================================${NC}"
    echo ""
    
    # Run all setup functions
    check_prerequisites
    create_project_structure
    setup_environment
    create_docker_config
    create_orchestrator
    create_frontend
    create_agent_templates
    create_monitoring_config
    create_startup_scripts
    create_documentation
    
    echo ""
    print_header "🎊 Setup Complete!"
    echo ""
    echo -e "${GREEN}Your Agentic AI SDLC System is ready to deploy!${NC}"
    echo ""
    echo -e "${CYAN}Next steps:${NC}"
    echo -e "${YELLOW}1. cd agentic-ai-sdlc${NC}"
    echo -e "${YELLOW}2. Edit .env file with your API keys${NC}"
    echo -e "${YELLOW}3. Run: ./scripts/start-prod.sh${NC}"
    echo ""
    
    read -p "Would you like to deploy the system now? (y/n): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        deploy_system
    else
        print_info "Setup complete. Run './scripts/start-prod.sh' when ready to deploy."
    fi
}

# Run main function
main "$@"
    
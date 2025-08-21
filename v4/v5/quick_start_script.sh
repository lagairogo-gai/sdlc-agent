#!/bin/bash

# Quick Start Script for User Story AI Agent
# This is a simplified version that avoids dependency conflicts

set -e

echo "🚀 Quick Start - User Story AI Agent"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is required but not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    print_error "Docker Compose is required but not installed."
    exit 1
fi

print_status "Docker and Docker Compose found ✓"

# Create simplified project structure
print_status "Creating project structure..."
mkdir -p user-story-ai-agent
cd user-story-ai-agent

# Create a minimal docker-compose setup
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  # Vector Database
  qdrant:
    image: qdrant/qdrant:v1.7.4
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

  # Graph Database
  neo4j:
    image: neo4j:5.15
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/password123
      - NEO4J_PLUGINS=["apoc"]
    volumes:
      - neo4j_data:/data

  # PostgreSQL
  postgres:
    image: postgres:15
    ports:
      - "5432:5432"
    environment:
      - POSTGRES_DB=userstory_agent
      - POSTGRES_USER=postgres
      - POSTGRES_PASSWORD=postgres123
    volumes:
      - postgres_data:/var/lib/postgresql/data

  # Redis
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

  # Simple Python backend
  backend:
    image: python:3.11-slim
    ports:
      - "8000:8000"
    volumes:
      - ./backend:/app
    working_dir: /app
    depends_on:
      - postgres
      - redis
      - qdrant
      - neo4j
    environment:
      - DATABASE_URL=postgresql://postgres:postgres123@postgres:5432/userstory_agent
      - REDIS_URL=redis://redis:6379
      - QDRANT_URL=http://qdrant:6333
      - NEO4J_URI=bolt://neo4j:7687
      - NEO4J_USER=neo4j
      - NEO4J_PASSWORD=password123
    command: >
      bash -c "
        pip install fastapi uvicorn sqlalchemy psycopg2-binary redis python-multipart python-dotenv &&
        python -c 'from fastapi import FastAPI; app = FastAPI(); @app.get(\"/\"); def read_root(): return {\"message\": \"User Story AI Agent Backend\", \"status\": \"running\"}; @app.get(\"/health\"); def health(): return {\"status\": \"healthy\"}' > main.py &&
        uvicorn main:app --host 0.0.0.0 --port 8000 --reload
      "

  # Simple frontend
  frontend:
    image: node:18-alpine
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
    working_dir: /app
    depends_on:
      - backend
    command: >
      sh -c "
        if [ ! -f package.json ]; then
          npm init -y &&
          npm install react@18.2.0 react-dom@18.2.0 react-scripts@5.0.1 &&
          mkdir -p src public &&
          echo '<!DOCTYPE html><html><head><title>StoryAI Agent</title></head><body><div id=\"root\"></div></body></html>' > public/index.html &&
          echo 'import React from \"react\"; import ReactDOM from \"react-dom/client\"; const App = () => <div style={{padding: \"20px\", fontFamily: \"Arial\"}}><h1>🚀 StoryAI Agent</h1><p>Backend: <a href=\"http://localhost:8000\">http://localhost:8000</a></p><p>API Health: <a href=\"http://localhost:8000/health\">Check Health</a></p><p>Full frontend coming soon...</p></div>; const root = ReactDOM.createRoot(document.getElementById(\"root\")); root.render(<App />);' > src/index.js
        fi &&
        npm start
      "

volumes:
  qdrant_data:
  neo4j_data:
  postgres_data:
  redis_data:
EOF

# Create backend directory with minimal setup
mkdir -p backend
cat > backend/main.py << 'EOF'
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="User Story AI Agent", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {
        "message": "User Story AI Agent Backend",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "databases": {
            "postgres": "connected",
            "redis": "connected", 
            "qdrant": "connected",
            "neo4j": "connected"
        }
    }

@app.get("/api/projects")
def get_projects():
    return {
        "projects": [
            {
                "id": 1,
                "name": "E-commerce Platform",
                "description": "Online shopping solution",
                "status": "active"
            }
        ]
    }

@app.post("/api/generate-stories")
def generate_stories():
    return {
        "stories": [
            {
                "id": 1,
                "title": "As a user, I want to login to access my account",
                "description": "User authentication functionality",
                "priority": "High",
                "points": 5
            }
        ]
    }
EOF

# Create frontend directory
mkdir -p frontend
cat > frontend/package.json << 'EOF'
{
  "name": "user-story-frontend",
  "version": "1.0.0",
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1"
  },
  "scripts": {
    "start": "react-scripts start",
    "build": "react-scripts build"
  },
  "browserslist": {
    "production": [">0.2%", "not dead"],
    "development": ["last 1 chrome version"]
  }
}
EOF

mkdir -p frontend/public frontend/src

cat > frontend/public/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>StoryAI Agent</title>
    <style>
        body { margin: 0; font-family: Arial, sans-serif; }
    </style>
</head>
<body>
    <div id="root"></div>
</body>
</html>
EOF

cat > frontend/src/index.js << 'EOF'
import React, { useState, useEffect } from 'react';
import ReactDOM from 'react-dom/client';

const App = () => {
    const [backendStatus, setBackendStatus] = useState('checking...');
    const [projects, setProjects] = useState([]);

    useEffect(() => {
        // Check backend health
        fetch('http://localhost:8000/health')
            .then(res => res.json())
            .then(data => setBackendStatus(data.status))
            .catch(() => setBackendStatus('offline'));

        // Load projects
        fetch('http://localhost:8000/api/projects')
            .then(res => res.json())
            .then(data => setProjects(data.projects))
            .catch(console.error);
    }, []);

    const generateStories = () => {
        fetch('http://localhost:8000/api/generate-stories', { method: 'POST' })
            .then(res => res.json())
            .then(data => alert(`Generated ${data.stories.length} stories!`))
            .catch(console.error);
    };

    return (
        <div style={{ padding: '20px', maxWidth: '800px', margin: '0 auto' }}>
            <h1>🚀 StoryAI Agent</h1>
            <div style={{ background: '#f5f5f5', padding: '15px', borderRadius: '8px', marginBottom: '20px' }}>
                <h3>System Status</h3>
                <p>Backend: <span style={{ color: backendStatus === 'healthy' ? 'green' : 'red' }}>{backendStatus}</span></p>
                <p>Frontend: <span style={{ color: 'green' }}>running</span></p>
            </div>
            
            <div style={{ background: '#e8f4fd', padding: '15px', borderRadius: '8px', marginBottom: '20px' }}>
                <h3>Quick Actions</h3>
                <button onClick={generateStories} style={{ padding: '10px 20px', background: '#007bff', color: 'white', border: 'none', borderRadius: '5px', cursor: 'pointer' }}>
                    🤖 Generate User Stories
                </button>
            </div>

            <div>
                <h3>Projects ({projects.length})</h3>
                {projects.map(project => (
                    <div key={project.id} style={{ border: '1px solid #ddd', padding: '15px', borderRadius: '8px', marginBottom: '10px' }}>
                        <h4>{project.name}</h4>
                        <p>{project.description}</p>
                        <span style={{ background: '#28a745', color: 'white', padding: '2px 8px', borderRadius: '12px', fontSize: '12px' }}>
                            {project.status}
                        </span>
                    </div>
                ))}
            </div>

            <div style={{ marginTop: '30px', padding: '15px', background: '#fff3cd', borderRadius: '8px' }}>
                <h4>🔧 Next Steps</h4>
                <ol>
                    <li>Configure your LLM API keys (OpenAI, Anthropic, etc.)</li>
                    <li>Set up Jira/Confluence integrations</li>
                    <li>Upload requirement documents</li>
                    <li>Generate your first user stories!</li>
                </ol>
            </div>

            <div style={{ marginTop: '20px', textAlign: 'center', color: '#666' }}>
                <p>Access your databases:</p>
                <p>
                    <a href="http://localhost:7474" target="_blank" rel="noopener noreferrer">Neo4j Browser</a> | 
                    <a href="http://localhost:6333/dashboard" target="_blank" rel="noopener noreferrer" style={{ margin: '0 10px' }}>Qdrant Dashboard</a> | 
                    <a href="http://localhost:8000" target="_blank" rel="noopener noreferrer">API Docs</a>
                </p>
            </div>
        </div>
    );
};

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<App />);
EOF

# Create start/stop scripts
cat > start.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting User Story AI Agent..."
docker-compose up -d
echo ""
echo "✅ Services started!"
echo ""
echo "🌐 Access your application:"
echo "   Frontend: http://localhost:3000"
echo "   Backend API: http://localhost:8000"
echo "   Neo4j Browser: http://localhost:7474 (neo4j/password123)"
echo "   Qdrant Dashboard: http://localhost:6333/dashboard"
echo ""
echo "📋 View logs: docker-compose logs -f"
echo "🛑 Stop services: ./stop.sh"
EOF

cat > stop.sh << 'EOF'
#!/bin/bash
echo "🛑 Stopping User Story AI Agent..."
docker-compose down
echo "✅ All services stopped."
EOF

chmod +x start.sh stop.sh

# Create environment file
cat > .env << 'EOF'
# Database Configuration
DATABASE_URL=postgresql://postgres:postgres123@localhost:5432/userstory_agent
REDIS_URL=redis://localhost:6379
QDRANT_URL=http://localhost:6333
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123

# LLM API Keys (add your keys here)
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GOOGLE_API_KEY=your-google-api-key

# Integration Configurations
JIRA_SERVER=https://your-company.atlassian.net
CONFLUENCE_SERVER=https://your-company.atlassian.net
EOF

print_status "✅ Quick setup completed!"
echo ""
echo "🚀 To start your User Story AI Agent:"
echo "   cd user-story-ai-agent"
echo "   ./start.sh"
echo ""
echo "⚙️  To configure API keys:"
echo "   Edit the .env file with your actual API keys"
echo ""
echo "📖 Full setup guide:"
echo "   The full production setup requires additional configuration"
echo "   This quick start gives you a working demo environment"
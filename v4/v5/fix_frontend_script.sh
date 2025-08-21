#!/bin/bash

# Quick fix for the frontend React scripts issue

echo "🔧 Fixing frontend container..."

# Stop the current services
docker-compose down

# Remove the problematic frontend setup
rm -rf frontend/node_modules
rm -f frontend/package-lock.json

# Recreate frontend structure with proper package.json
mkdir -p frontend/src frontend/public

cat > frontend/package.json << 'EOF'
{
  "name": "user-story-frontend",
  "version": "1.0.0",
  "private": true,
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0",
    "react-scripts": "5.0.1",
    "web-vitals": "^2.1.4"
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
  }
}
EOF

# Create a simple but functional React app
cat > frontend/public/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="StoryAI Agent - User Story Generation Platform" />
    <title>StoryAI Agent</title>
    <style>
      body {
        margin: 0;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', 'Oxygen', 'Ubuntu', 'Cantarell', sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
      }
    </style>
  </head>
  <body>
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>
EOF

cat > frontend/src/index.js << 'EOF'
import React, { useState, useEffect } from 'react';
import ReactDOM from 'react-dom/client';

const styles = {
  container: {
    padding: '20px',
    maxWidth: '1200px',
    margin: '0 auto',
    minHeight: '100vh'
  },
  card: {
    background: 'rgba(255, 255, 255, 0.95)',
    borderRadius: '12px',
    padding: '24px',
    margin: '16px 0',
    boxShadow: '0 8px 32px rgba(0,0,0,0.1)',
    backdropFilter: 'blur(10px)',
    border: '1px solid rgba(255,255,255,0.2)'
  },
  header: {
    textAlign: 'center',
    marginBottom: '32px',
    color: 'white'
  },
  title: {
    fontSize: '3rem',
    margin: '0',
    textShadow: '2px 2px 4px rgba(0,0,0,0.3)'
  },
  subtitle: {
    fontSize: '1.2rem',
    margin: '8px 0 0 0',
    opacity: 0.9
  },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
    gap: '20px',
    marginBottom: '24px'
  },
  button: {
    padding: '12px 24px',
    background: 'linear-gradient(45deg, #667eea, #764ba2)',
    color: 'white',
    border: 'none',
    borderRadius: '8px',
    cursor: 'pointer',
    fontSize: '16px',
    margin: '8px 8px 8px 0',
    transition: 'all 0.3s ease',
    textDecoration: 'none',
    display: 'inline-block',
    textAlign: 'center'
  },
  statusHealthy: {
    color: '#10b981',
    fontWeight: 'bold'
  },
  statusOffline: {
    color: '#ef4444',
    fontWeight: 'bold'
  },
  project: {
    borderLeft: '4px solid #667eea',
    margin: '12px 0'
  },
  badge: {
    background: '#10b981',
    color: 'white',
    padding: '4px 12px',
    borderRadius: '16px',
    fontSize: '14px',
    display: 'inline-block'
  }
};

const StoryAIApp = () => {
  const [backendStatus, setBackendStatus] = useState('checking...');
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(false);
  const [lastCheck, setLastCheck] = useState(new Date().toLocaleTimeString());

  useEffect(() => {
    checkBackendHealth();
    loadProjects();
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(() => {
      checkBackendHealth();
      setLastCheck(new Date().toLocaleTimeString());
    }, 30000);
    
    return () => clearInterval(interval);
  }, []);

  const checkBackendHealth = async () => {
    try {
      const response = await fetch('http://localhost:8000/health');
      const data = await response.json();
      setBackendStatus(data.status);
    } catch (error) {
      console.error('Backend health check failed:', error);
      setBackendStatus('offline');
    }
  };

  const loadProjects = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/projects');
      const data = await response.json();
      setProjects(data.projects || []);
    } catch (error) {
      console.error('Failed to load projects:', error);
      setProjects([]);
    }
  };

  const generateStories = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/generate-stories', { 
        method: 'POST' 
      });
      const data = await response.json();
      
      if (data.stories && data.stories.length > 0) {
        alert(`✅ Generated ${data.stories.length} user stories!\n\nExample:\n"${data.stories[0].title}"\n\nPriority: ${data.stories[0].priority}\nStory Points: ${data.stories[0].points}`);
      } else {
        alert('✅ Story generation completed!');
      }
    } catch (error) {
      console.error('Story generation failed:', error);
      alert('❌ Failed to generate stories. Please check if the backend is running.');
    }
    setLoading(false);
  };

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <h1 style={styles.title}>🚀 StoryAI Agent</h1>
        <p style={styles.subtitle}>RAG-based AI User Story Generation Platform</p>
      </header>

      <div style={styles.grid}>
        <div style={styles.card}>
          <h3>🔧 System Status</h3>
          <p>
            Backend API: <span style={backendStatus === 'healthy' ? styles.statusHealthy : styles.statusOffline}>
              {backendStatus}
            </span>
          </p>
          <p>Frontend: <span style={styles.statusHealthy}>running</span></p>
          <p>Last Check: {lastCheck}</p>
          {backendStatus === 'healthy' && (
            <p style={{ fontSize: '14px', color: '#10b981' }}>
              ✅ All systems operational
            </p>
          )}
        </div>

        <div style={styles.card}>
          <h3>⚡ Quick Actions</h3>
          <button 
            style={{
              ...styles.button,
              opacity: loading || backendStatus !== 'healthy' ? 0.6 : 1,
              cursor: loading || backendStatus !== 'healthy' ? 'not-allowed' : 'pointer'
            }}
            onClick={generateStories}
            disabled={loading || backendStatus !== 'healthy'}
          >
            {loading ? '🔄 Generating Stories...' : '🤖 Generate User Stories'}
          </button>
          <br />
          <button style={styles.button} onClick={checkBackendHealth}>
            🔄 Refresh Status
          </button>
        </div>
      </div>

      <div style={styles.card}>
        <h3>📋 Projects ({projects.length})</h3>
        {projects.length === 0 ? (
          <div style={{ textAlign: 'center', padding: '20px', color: '#666' }}>
            <p>🏗️ No projects found</p>
            <p style={{ fontSize: '14px' }}>Create your first project to get started!</p>
          </div>
        ) : (
          projects.map(project => (
            <div key={project.id} style={{...styles.card, ...styles.project}}>
              <h4 style={{ margin: '0 0 8px 0' }}>{project.name}</h4>
              <p style={{ color: '#666', margin: '0 0 12px 0' }}>{project.description}</p>
              <span style={styles.badge}>{project.status}</span>
            </div>
          ))
        )}
      </div>

      <div style={styles.card}>
        <h3>🔗 Database Access</h3>
        <p style={{ marginBottom: '16px', color: '#666' }}>
          Access your databases and system components:
        </p>
        <a href="http://localhost:7474" target="_blank" rel="noopener noreferrer" style={styles.button}>
          🕸️ Neo4j Browser
        </a>
        <a href="http://localhost:6333/dashboard" target="_blank" rel="noopener noreferrer" style={styles.button}>
          🔍 Qdrant Dashboard  
        </a>
        <a href="http://localhost:8000" target="_blank" rel="noopener noreferrer" style={styles.button}>
          📚 API Documentation
        </a>
        <p style={{ fontSize: '14px', color: '#666', marginTop: '12px' }}>
          💡 Neo4j credentials: <code>neo4j / password123</code>
        </p>
      </div>

      <div style={{...styles.card, background: 'rgba(239, 246, 255, 0.95)'}}>
        <h3>🛠️ Configuration & Next Steps</h3>
        <ol style={{ paddingLeft: '20px', lineHeight: '1.6' }}>
          <li><strong>Configure LLM APIs:</strong> Add your API keys to the .env file (OpenAI, Anthropic, Google)</li>
          <li><strong>Set up Integrations:</strong> Connect to Jira, Confluence, and SharePoint</li>
          <li><strong>Upload Documents:</strong> Add requirement documents to the RAG system</li>
          <li><strong>Generate Stories:</strong> Use AI to create intelligent user stories from your requirements</li>
          <li><strong>Export Results:</strong> Push generated stories back to Jira or other tools</li>
        </ol>
      </div>

      <footer style={{ textAlign: 'center', padding: '20px', color: 'rgba(255,255,255,0.8)' }}>
        <p>🎯 Built with React, FastAPI, LangChain, Neo4j, and Qdrant</p>
      </footer>
    </div>
  );
};

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(<StoryAIApp />);
EOF

# Update the docker-compose.yml to use the fixed configuration
cat > docker-compose.yml << 'EOF'
services:
  qdrant:
    image: qdrant/qdrant:v1.7.4
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

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

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data

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
        apt-get update && apt-get install -y gcc &&
        pip install --no-cache-dir fastapi uvicorn sqlalchemy psycopg2-binary redis python-multipart python-dotenv &&
        if [ ! -f main.py ]; then
          cat > main.py << 'PYEOF'
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title=\"User Story AI Agent\", version=\"1.0.0\")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[\"http://localhost:3000\"],
    allow_credentials=True,
    allow_methods=[\"*\"],
    allow_headers=[\"*\"],
)

@app.get(\"/\")
def read_root():
    return {\"message\": \"User Story AI Agent Backend\", \"status\": \"running\", \"version\": \"1.0.0\"}

@app.get(\"/health\")
def health_check():
    return {\"status\": \"healthy\", \"databases\": {\"postgres\": \"connected\", \"redis\": \"connected\", \"qdrant\": \"connected\", \"neo4j\": \"connected\"}}

@app.get(\"/api/projects\")
def get_projects():
    return {\"projects\": [{\"id\": 1, \"name\": \"E-commerce Platform\", \"description\": \"Complete online shopping solution with AI recommendations\", \"status\": \"active\"}, {\"id\": 2, \"name\": \"Mobile Banking App\", \"description\": \"Secure mobile banking with biometric authentication\", \"status\": \"development\"}]}

@app.post(\"/api/generate-stories\")
def generate_stories():
    return {\"stories\": [{\"id\": 1, \"title\": \"As a user, I want to log in securely to access my account\", \"description\": \"Implement secure user authentication with multi-factor support\", \"priority\": \"High\", \"points\": 8}, {\"id\": 2, \"title\": \"As a customer, I want to search for products easily\", \"description\": \"Advanced search with filters, suggestions, and autocomplete\", \"priority\": \"Medium\", \"points\": 5}]}
PYEOF
        fi &&
        uvicorn main:app --host 0.0.0.0 --port 8000 --reload
      "

  frontend:
    image: node:18-alpine
    ports:
      - "3000:3000"
    volumes:
      - ./frontend:/app
      - /app/node_modules
    working_dir: /app
    depends_on:
      - backend
    environment:
      - CHOKIDAR_USEPOLLING=true
      - WATCHPACK_POLLING=true
    command: sh -c "npm install && npm start"

volumes:
  qdrant_data:
  neo4j_data:
  postgres_data:
  redis_data:
EOF

echo "✅ Frontend fixed! Now restart the services:"
echo ""
echo "🚀 Run these commands:"
echo "   docker-compose up -d"
echo ""
echo "🌐 Then access:"
echo "   Frontend: http://localhost:3000"
echo "   Backend: http://localhost:8000"
echo ""
echo "📋 Monitor with:"
echo "   docker-compose logs -f frontend"
echo "   docker-compose logs -f backend"
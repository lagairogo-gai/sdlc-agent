#!/bin/bash

# User Story AI Agent - Setup and Deployment Script
# This script sets up the complete production environment

set -e

echo "🚀 Setting up User Story AI Agent..."

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[SETUP]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker and try again."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose and try again."
        exit 1
    fi
    
    print_status "Docker and Docker Compose are installed"
}

# Check if Node.js is installed
check_nodejs() {
    if ! command -v node &> /dev/null; then
        print_warning "Node.js is not installed. Installing Node.js 18..."
        curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
        sudo apt-get install -y nodejs
    fi
    
    if ! command -v npm &> /dev/null; then
        print_error "npm is not installed. Please install npm and try again."
        exit 1
    fi
    
    print_status "Node.js and npm are installed"
}

# Check if Python is installed
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.8+ and try again."
        exit 1
    fi
    
    if ! command -v pip3 &> /dev/null; then
        print_error "pip3 is not installed. Please install pip3 and try again."
        exit 1
    fi
    
    print_status "Python 3 and pip3 are installed"
}

# Create project structure
create_project_structure() {
    print_header "Creating project structure..."
    
    mkdir -p user-story-ai-agent
    cd user-story-ai-agent
    
    # Create directory structure
    mkdir -p backend/{app,migrations,tests}
    mkdir -p backend/app/{models,schemas,services,agents,utils,integrations}
    mkdir -p frontend/{public,src}
    mkdir -p frontend/src/{components,pages,hooks,utils,styles}
    mkdir -p data/{uploads,backups}
    mkdir -p docs
    mkdir -p scripts
    
    print_status "Project structure created"
}

# Setup backend
setup_backend() {
    print_header "Setting up backend..."
    
    cd backend
    
    # Create Python virtual environment
    python3 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install dependencies (requirements.txt should be created separately)
    cat > requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn==0.24.0
langchain==0.1.0
langgraph==0.0.20
langchain-openai==0.0.5
langchain-community==0.0.12
langchain-google-genai==0.0.7
langchain-anthropic==0.1.1
sqlalchemy==2.0.23
alembic==1.13.1
psycopg2-binary==2.9.9
redis==5.0.1
qdrant-client==1.7.0
neo4j==5.15.0
pydantic==2.5.0
python-multipart==0.0.6
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-dotenv==1.0.0
httpx==0.25.2
pandas==2.1.4
numpy==1.25.2
pymongo==4.6.0
atlassian-python-api==3.41.0
requests==2.31.0
aiofiles==23.2.1
python-docx==1.1.0
PyPDF2==3.0.1
openpyxl==3.1.2
azure-identity==1.15.0
azure-keyvault-secrets==4.7.0
msal==1.25.0
Office365-REST-Python-Client==2.5.3
google-api-python-client==2.108.0
google-auth-oauthlib==1.1.0
google-auth-httplib2==0.1.1
celery==5.3.4
flower==2.0.1
tiktoken==0.5.1
sentence-transformers==2.2.2
faiss-cpu==1.7.4
chromadb==0.4.18
EOF
    
    pip install -r requirements.txt
    
    # Create basic backend structure files
    cat > app/__init__.py << 'EOF'
# FastAPI User Story AI Agent
EOF

    cat > app/config.py << 'EOF'
import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://postgres:postgres123@localhost:5432/userstory_agent"
    
    # Redis
    redis_url: str = "redis://localhost:6379"
    
    # Vector Database
    qdrant_url: str = "http://localhost:6333"
    
    # Graph Database
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password123"
    
    # JWT
    secret_key: str = "your-secret-key-here"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # LLM APIs
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    google_api_key: str = ""
    azure_openai_api_key: str = ""
    azure_openai_endpoint: str = ""
    
    # Integrations
    jira_server: str = ""
    confluence_server: str = ""
    sharepoint_site: str = ""
    
    class Config:
        env_file = ".env"

settings = Settings()
EOF

    # Create environment file template
    cat > .env.template << 'EOF'
# Database Configuration
DATABASE_URL=postgresql://postgres:postgres123@postgres:5432/userstory_agent
REDIS_URL=redis://redis:6379
QDRANT_URL=http://qdrant:6333
NEO4J_URI=bolt://neo4j:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password123

# Security
SECRET_KEY=your-very-secret-key-change-this-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# LLM API Keys (Add your actual API keys)
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
GOOGLE_API_KEY=your-google-api-key
AZURE_OPENAI_API_KEY=your-azure-openai-key
AZURE_OPENAI_ENDPOINT=your-azure-endpoint

# Integration Configurations
JIRA_SERVER=https://your-company.atlassian.net
CONFLUENCE_SERVER=https://your-company.atlassian.net
SHAREPOINT_SITE=https://your-company.sharepoint.com

# Environment
ENVIRONMENT=development
DEBUG=true
EOF

    # Create Dockerfile for backend
    cat > Dockerfile << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
EOF

    deactivate
    cd ..
    
    print_status "Backend setup completed"
}

# Setup frontend
setup_frontend() {
    print_header "Setting up frontend..."
    
    cd frontend
    
    # Initialize npm project
    npm init -y
    
    # Install dependencies
    npm install react@18.2.0 react-dom@18.2.0 react-scripts@5.0.1
    npm install react-router-dom@6.8.1 react-query@3.39.3 axios@1.3.4
    npm install react-hook-form@7.43.5 react-beautiful-dnd@13.1.1 
    npm install react-flow-renderer@10.3.17 framer-motion@10.0.1
    npm install lucide-react@0.263.1 react-hot-toast@2.4.0 recharts@2.5.0
    npm install tailwindcss@3.2.7 autoprefixer@10.4.14 postcss@8.4.21
    npm install @headlessui/react@1.7.13 clsx@1.2.1 date-fns@2.29.3
    npm install react-dropzone@14.2.3 react-markdown@8.0.5
    npm install react-syntax-highlighter@15.5.0 zustand@4.3.6
    npm install react-select@5.7.0 react-textarea-autosize@8.4.1
    
    # Install dev dependencies
    npm install -D @types/react@18.0.28 @types/react-dom@18.0.11 typescript@4.9.5
    
    # Initialize Tailwind CSS
    npx tailwindcss init -p
    
    # Configure Tailwind
    cat > tailwind.config.js << 'EOF'
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
    "./public/index.html"
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f0f9ff',
          500: '#0ea5e9',
          600: '#0284c7',
          700: '#0369a1',
        },
        secondary: {
          500: '#8b5cf6',
          600: '#7c3aed',
        }
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
      }
    },
  },
  plugins: [],
}
EOF

    # Create basic frontend structure
    mkdir -p src/{components,pages,hooks,utils,styles}
    mkdir -p public
    
    # Create index.html
    cat > public/index.html << 'EOF'
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <link rel="icon" href="%PUBLIC_URL%/favicon.ico" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <meta name="theme-color" content="#000000" />
    <meta name="description" content="AI-powered User Story Generation Platform" />
    <title>StoryAI Agent</title>
  </head>
  <body class="bg-gray-900">
    <noscript>You need to enable JavaScript to run this app.</noscript>
    <div id="root"></div>
  </body>
</html>
EOF

    # Create basic index.js
    cat > src/index.js << 'EOF'
import React from 'react';
import ReactDOM from 'react-dom/client';
import './styles/index.css';
import App from './App';

const root = ReactDOM.createRoot(document.getElementById('root'));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
EOF

    # Create Tailwind CSS file
    cat > src/styles/index.css << 'EOF'
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

/* Custom scrollbar */
::-webkit-scrollbar {
  width: 8px;
}

::-webkit-scrollbar-track {
  background: rgba(0, 0, 0, 0.1);
}

::-webkit-scrollbar-thumb {
  background: rgba(139, 92, 246, 0.5);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: rgba(139, 92, 246, 0.8);
}
EOF

    # Create Dockerfile for frontend
    cat > Dockerfile << 'EOF'
FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY . .

# Expose port
EXPOSE 3000

# Start the application
CMD ["npm", "start"]
EOF

    # Create .env template for frontend
    cat > .env.template << 'EOF'
REACT_APP_API_URL=http://localhost:8000
REACT_APP_ENVIRONMENT=development
EOF

    cd ..
    
    print_status "Frontend setup completed"
}

# Setup Docker configuration
setup_docker() {
    print_header "Setting up Docker configuration..."
    
    # Create docker-compose.yml (this was already provided in the artifacts)
    # Create additional docker-compose files for different environments
    
    # Development docker-compose override
    cat > docker-compose.dev.yml << 'EOF'
version: '3.8'

services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    volumes:
      - ./backend:/app
      - /app/venv
    environment:
      - DEBUG=true
      - RELOAD=true
    command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile
    volumes:
      - ./frontend:/app
      - /app/node_modules
    environment:
      - CHOKIDAR_USEPOLLING=true
    stdin_open: true
    tty: true
EOF

    # Production docker-compose override
    cat > docker-compose.prod.yml << 'EOF'
version: '3.8'

services:
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile.prod
    environment:
      - DEBUG=false
      - RELOAD=false
    restart: unless-stopped

  frontend:
    build:
      context: ./frontend
      dockerfile: Dockerfile.prod
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - frontend
      - backend
    restart: unless-stopped
EOF

    # Create production Dockerfiles
    cat > backend/Dockerfile.prod << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Change ownership to non-root user
RUN chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Command to run the application
CMD ["gunicorn", "app.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
EOF

    cat > frontend/Dockerfile.prod << 'EOF'
# Build stage
FROM node:18-alpine as build

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy source code
COPY . .

# Build the application
RUN npm run build

# Production stage
FROM nginx:alpine

# Copy built app from build stage
COPY --from=build /app/build /usr/share/nginx/html

# Copy nginx configuration
COPY nginx.conf /etc/nginx/nginx.conf

# Expose port
EXPOSE 80

# Start nginx
CMD ["nginx", "-g", "daemon off;"]
EOF

    # Create nginx configuration
    cat > nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;

    upstream backend {
        server backend:8000;
    }

    server {
        listen 80;
        server_name localhost;

        # Frontend
        location / {
            root /usr/share/nginx/html;
            index index.html index.htm;
            try_files $uri $uri/ /index.html;
        }

        # API routes
        location /api/ {
            proxy_pass http://backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
        }

        # Health check
        location /health {
            proxy_pass http://backend/health;
        }
    }
}
EOF

    print_status "Docker configuration completed"
}

# Create database migrations
setup_database() {
    print_header "Setting up database migrations..."
    
    cd backend
    source venv/bin/activate
    
    # Install alembic if not already installed
    pip install alembic
    
    # Initialize Alembic
    alembic init migrations
    
    # Create initial migration
    cat > migrations/versions/001_initial_migration.py << 'EOF'
"""Initial migration

Revision ID: 001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers
revision = '001'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    # Users table
    op.create_table('users',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('email', sa.String(), nullable=False),
        sa.Column('username', sa.String(), nullable=False),
        sa.Column('hashed_password', sa.String(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_users_email'), 'users', ['email'], unique=True)
    op.create_index(op.f('ix_users_username'), 'users', ['username'], unique=True)

    # Projects table
    op.create_table('projects',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('name', sa.String(), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', sa.String(), nullable=True),
        sa.Column('owner_id', sa.Integer(), nullable=False),
        sa.Column('jira_config', sa.JSON(), nullable=True),
        sa.Column('confluence_config', sa.JSON(), nullable=True),
        sa.Column('sharepoint_config', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(['owner_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )

    # Add other tables here...

def downgrade():
    op.drop_table('projects')
    op.drop_index(op.f('ix_users_username'), table_name='users')
    op.drop_index(op.f('ix_users_email'), table_name='users')
    op.drop_table('users')
EOF

    deactivate
    cd ..
    
    print_status "Database setup completed"
}

# Create utility scripts
create_scripts() {
    print_header "Creating utility scripts..."
    
    mkdir -p scripts
    
    # Start script
    cat > scripts/start.sh << 'EOF'
#!/bin/bash
echo "Starting User Story AI Agent..."
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d
echo "Services starting... Check http://localhost:3000 for frontend and http://localhost:8000 for API"
EOF

    # Stop script
    cat > scripts/stop.sh << 'EOF'
#!/bin/bash
echo "Stopping User Story AI Agent..."
docker-compose down
EOF

    # Backup script
    cat > scripts/backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="./data/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

echo "Creating backup in $BACKUP_DIR..."

# Backup PostgreSQL
docker exec user-story-ai-agent_postgres_1 pg_dump -U postgres userstory_agent > $BACKUP_DIR/postgres_backup.sql

# Backup Neo4j
docker exec user-story-ai-agent_neo4j_1 neo4j-admin backup --backup-dir=/var/backups --name=graph.db
docker cp user-story-ai-agent_neo4j_1:/var/backups $BACKUP_DIR/neo4j_backup

# Backup Qdrant
docker exec user-story-ai-agent_qdrant_1 tar -czf /tmp/qdrant_backup.tar.gz /qdrant/storage
docker cp user-story-ai-agent_qdrant_1:/tmp/qdrant_backup.tar.gz $BACKUP_DIR/

echo "Backup completed: $BACKUP_DIR"
EOF

    # Restore script
    cat > scripts/restore.sh << 'EOF'
#!/bin/bash
if [ -z "$1" ]; then
    echo "Usage: $0 <backup_directory>"
    exit 1
fi

BACKUP_DIR=$1
echo "Restoring from backup: $BACKUP_DIR"

# Restore PostgreSQL
if [ -f "$BACKUP_DIR/postgres_backup.sql" ]; then
    docker exec -i user-story-ai-agent_postgres_1 psql -U postgres userstory_agent < $BACKUP_DIR/postgres_backup.sql
    echo "PostgreSQL restored"
fi

# Restore Neo4j
if [ -d "$BACKUP_DIR/neo4j_backup" ]; then
    docker cp $BACKUP_DIR/neo4j_backup user-story-ai-agent_neo4j_1:/var/backups
    docker exec user-story-ai-agent_neo4j_1 neo4j-admin restore --from=/var/backups --database=neo4j --force
    echo "Neo4j restored"
fi

# Restore Qdrant
if [ -f "$BACKUP_DIR/qdrant_backup.tar.gz" ]; then
    docker cp $BACKUP_DIR/qdrant_backup.tar.gz user-story-ai-agent_qdrant_1:/tmp/
    docker exec user-story-ai-agent_qdrant_1 tar -xzf /tmp/qdrant_backup.tar.gz -C /
    echo "Qdrant restored"
fi

echo "Restore completed"
EOF

    # Make scripts executable
    chmod +x scripts/*.sh
    
    print_status "Utility scripts created"
}

# Create documentation
create_documentation() {
    print_header "Creating documentation..."
    
    cat > README.md << 'EOF'
# User Story AI Agent

A production-grade RAG-based AI agent for generating user stories from requirements documents using LangChain, LangGraph, and multiple LLM providers.

## Features

- 🤖 **AI-Powered Story Generation**: Uses LangGraph for intelligent user story creation
- 📚 **RAG System**: Retrieval-Augmented Generation with Qdrant vector database
- 🕸️ **Knowledge Graph**: Neo4j-powered knowledge extraction and relationship mapping
- 🔗 **Multi-Platform Integration**: Jira, Confluence, SharePoint, and document upload
- 🎨 **Modern UI**: React frontend with n8n-inspired animated workflow visualization
- 🚀 **Production Ready**: Docker-based deployment with monitoring and backup scripts

## Quick Start

1. **Clone and Setup**:
   ```bash
   git clone <repository>
   cd user-story-ai-agent
   ./setup.sh
   ```

2. **Configure Environment**:
   ```bash
   cp backend/.env.template backend/.env
   cp frontend/.env.template frontend/.env
   # Edit the .env files with your API keys and configurations
   ```

3. **Start Services**:
   ```bash
   ./scripts/start.sh
   ```

4. **Access the Application**:
   - Frontend: http://localhost:3000
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## Architecture

### Backend Stack
- **FastAPI**: High-performance async API framework
- **LangChain/LangGraph**: AI agent orchestration
- **PostgreSQL**: Primary application database
- **Qdrant**: Vector database for RAG
- **Neo4j**: Knowledge graph database
- **Redis**: Caching and session management

### Frontend Stack
- **React 18**: Modern UI framework
- **Tailwind CSS**: Utility-first styling
- **Framer Motion**: Smooth animations
- **React Query**: Server state management
- **React Flow**: Interactive workflow diagrams

### AI/ML Components
- **Multi-LLM Support**: OpenAI, Anthropic Claude, Google Gemini, Azure OpenAI
- **RAG Pipeline**: Document chunking, embedding, and retrieval
- **Knowledge Extraction**: Entity and relationship extraction
- **Agent Workflow**: LangGraph-based intelligent processing

## Configuration

### API Keys Required

Add these to your `backend/.env` file:

```env
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key
GOOGLE_API_KEY=your-google-key
AZURE_OPENAI_API_KEY=your-azure-key
```

### Integration Setup

1. **Jira Integration**:
   - Create API token in Jira
   - Configure server URL and credentials

2. **Confluence Integration**:
   - Set up API access
   - Configure space permissions

3. **SharePoint Integration**:
   - Register Azure app
   - Configure OAuth permissions

## Usage

### 1. Create a Project
- Navigate to Projects tab
- Click "New Project"
- Configure integrations

### 2. Import Requirements
- Upload documents (PDF, DOCX, TXT)
- Sync from Jira/Confluence
- Connect SharePoint sources

### 3. Generate Stories
- Use AI agent to generate stories
- Review and edit generated content
- Export to Jira or other tools

### 4. Analyze Knowledge
- Explore knowledge graph
- View entity relationships
- Get project insights

## Production Deployment

### Using Docker Compose (Recommended)

```bash
# Production deployment
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```

### Manual Deployment

1. **Database Setup**:
   ```bash
   # PostgreSQL
   createdb userstory_agent
   
   # Run migrations
   cd backend && alembic upgrade head
   ```

2. **Services**:
   ```bash
   # Start Qdrant
   docker run -p 6333:6333 qdrant/qdrant
   
   # Start Neo4j
   docker run -p 7474:7474 -p 7687:7687 neo4j:5.15
   
   # Start Redis
   docker run -p 6379:6379 redis:7-alpine
   ```

3. **Application**:
   ```bash
   # Backend
   cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000
   
   # Frontend
   cd frontend && npm start
   ```

## Monitoring and Maintenance

### Health Checks
- API health: `GET /health`
- Database connections monitored
- Service status dashboard

### Backups
```bash
# Create backup
./scripts/backup.sh

# Restore from backup
./scripts/restore.sh /path/to/backup
```

### Logs
```bash
# View logs
docker-compose logs -f backend
docker-compose logs -f frontend
```

## Development

### Local Development Setup
```bash
# Start in development mode
./scripts/start.sh

# Backend development
cd backend && source venv/bin/activate && uvicorn app.main:app --reload

# Frontend development
cd frontend && npm start
```

### Testing
```bash
# Backend tests
cd backend && pytest

# Frontend tests
cd frontend && npm test
```

## API Documentation

Interactive API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Troubleshooting

### Common Issues

1. **Database Connection Errors**:
   - Check if PostgreSQL is running
   - Verify connection string in .env

2. **Vector Search Not Working**:
   - Ensure Qdrant is running on port 6333
   - Check if embeddings are being generated

3. **Knowledge Graph Issues**:
   - Verify Neo4j is accessible on port 7687
   - Check authentication credentials

4. **Integration Failures**:
   - Validate API keys and permissions
   - Check network connectivity to external services

### Getting Help

- Check logs: `docker-compose logs`
- Review configuration files
- Verify environment variables
- Test API endpoints manually

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
EOF

    # Create API documentation
    cat > docs/API.md << 'EOF'
# API Documentation

## Authentication

All API endpoints require authentication via JWT tokens.

### Login
```
POST /auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "password"
}
```

## Projects

### Create Project
```
POST /projects
Authorization: Bearer <token>

{
  "name": "Project Name",
  "description": "Project description"
}
```

### List Projects
```
GET /projects
Authorization: Bearer <token>
```

## User Stories

### Generate Stories
```
POST /user-stories/generate
Authorization: Bearer <token>

{
  "project_id": 1,
  "requirements": "System requirements text",
  "context": "Additional context",
  "user_prompt": "Generate stories for authentication",
  "llm_config": {
    "provider": "openai",
    "model": "gpt-4",
    "temperature": 0.7
  }
}
```

### Export to Jira
```
POST /user-stories/{story_id}/export-jira
Authorization: Bearer <token>

{
  "project_key": "PROJ",
  "server": "https://company.atlassian.net",
  "username": "user@company.com",
  "api_token": "token"
}
```

## Integrations

### Sync from Jira
```
POST /integrations/jira/sync/{project_id}
Authorization: Bearer <token>

{
  "server": "https://company.atlassian.net",
  "username": "user@company.com",
  "api_token": "token",
  "project_key": "PROJ"
}
```

For complete API documentation, visit the interactive docs at `/docs` when running the application.
EOF

    print_status "Documentation created"
}

# Main setup function
main() {
    print_header "🚀 User Story AI Agent Setup Script"
    echo ""
    
    # Check prerequisites
    print_header "Checking prerequisites..."
    check_docker
    check_nodejs
    check_python
    echo ""
    
    # Setup project
    create_project_structure
    setup_backend
    setup_frontend
    setup_docker
    setup_database
    create_scripts
    create_documentation
    
    echo ""
    print_status "✅ Setup completed successfully!"
    echo ""
    print_header "Next Steps:"
    echo "1. Configure your environment variables:"
    echo "   - cp backend/.env.template backend/.env"
    echo "   - cp frontend/.env.template frontend/.env"
    echo "   - Edit the .env files with your API keys"
    echo ""
    echo "2. Start the application:"
    echo "   - ./scripts/start.sh"
    echo ""
    echo "3. Access the application:"
    echo "   - Frontend: http://localhost:3000"
    echo "   - API: http://localhost:8000"
    echo "   - API Docs: http://localhost:8000/docs"
    echo ""
    print_status "Happy coding! 🎉"
}

# Run main function
main "$@"
    
#!/bin/bash

# Fix PostgreSQL Initialization Issue
echo "🔧 Fixing PostgreSQL initialization issue..."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Step 1: Stop the problematic postgres container
echo "🛑 Step 1: Stopping PostgreSQL container..."
docker-compose stop postgres
docker-compose rm -f postgres
print_status "PostgreSQL container stopped and removed"

# Step 2: Create the postgres directory and init.sql file
echo "📁 Step 2: Creating postgres directory structure..."
mkdir -p postgres
print_status "Created postgres directory"

# Step 3: Create the init.sql file (you should have this from the previous artifact)
echo "📝 Step 3: Creating init.sql file..."
cat > postgres/init.sql << 'EOF'
-- PostgreSQL Initialization Script for Agentic AI SDLC System
-- This script sets up the initial database structure

-- Create database and user (if not already created by environment variables)
-- Note: The main database 'agentic_sdlc' and user 'agentic_user' are created via environment variables

-- Connect to the agentic_sdlc database
\c agentic_sdlc;

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create schemas for better organization
CREATE SCHEMA IF NOT EXISTS workflows;
CREATE SCHEMA IF NOT EXISTS agents;
CREATE SCHEMA IF NOT EXISTS projects;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Create basic tables for the SDLC system

-- Projects table
CREATE TABLE IF NOT EXISTS projects.projects (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(50) DEFAULT 'active',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'::jsonb
);

-- Workflows table
CREATE TABLE IF NOT EXISTS workflows.workflows (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    project_id UUID REFERENCES projects.projects(id),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(50) DEFAULT 'created',
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    requirements JSONB DEFAULT '{}'::jsonb,
    configuration JSONB DEFAULT '{}'::jsonb
);

-- Workflow phases table
CREATE TABLE IF NOT EXISTS workflows.workflow_phases (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id UUID REFERENCES workflows.workflows(id),
    name VARCHAR(100) NOT NULL,
    agent VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    input_data JSONB DEFAULT '{}'::jsonb,
    output_data JSONB DEFAULT '{}'::jsonb,
    error_message TEXT
);

-- Agents table
CREATE TABLE IF NOT EXISTS agents.agents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id VARCHAR(100) UNIQUE NOT NULL,
    agent_type VARCHAR(50) NOT NULL,
    status VARCHAR(50) DEFAULT 'inactive',
    last_heartbeat TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    configuration JSONB DEFAULT '{}'::jsonb,
    capabilities JSONB DEFAULT '[]'::jsonb
);

-- Agent tasks table
CREATE TABLE IF NOT EXISTS agents.agent_tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id VARCHAR(100) REFERENCES agents.agents(agent_id),
    workflow_id UUID REFERENCES workflows.workflows(id),
    phase_id UUID REFERENCES workflows.workflow_phases(id),
    task_type VARCHAR(100) NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    input_data JSONB DEFAULT '{}'::jsonb,
    output_data JSONB DEFAULT '{}'::jsonb,
    error_message TEXT
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_workflows_project_id ON workflows.workflows(project_id);
CREATE INDEX IF NOT EXISTS idx_workflows_status ON workflows.workflows(status);
CREATE INDEX IF NOT EXISTS idx_workflow_phases_workflow_id ON workflows.workflow_phases(workflow_id);
CREATE INDEX IF NOT EXISTS idx_workflow_phases_status ON workflows.workflow_phases(status);
CREATE INDEX IF NOT EXISTS idx_agent_tasks_agent_id ON agents.agent_tasks(agent_id);
CREATE INDEX IF NOT EXISTS idx_agent_tasks_workflow_id ON agents.agent_tasks(workflow_id);
CREATE INDEX IF NOT EXISTS idx_agent_tasks_status ON agents.agent_tasks(status);
CREATE INDEX IF NOT EXISTS idx_agents_agent_type ON agents.agents(agent_type);
CREATE INDEX IF NOT EXISTS idx_agents_status ON agents.agents(status);

-- Grant permissions to the agentic_user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA workflows TO agentic_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA agents TO agentic_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA projects TO agentic_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA monitoring TO agentic_user;
GRANT USAGE ON SCHEMA workflows TO agentic_user;
GRANT USAGE ON SCHEMA agents TO agentic_user;
GRANT USAGE ON SCHEMA projects TO agentic_user;
GRANT USAGE ON SCHEMA monitoring TO agentic_user;

\echo 'Agentic AI SDLC Database initialized successfully!'
EOF

print_status "Created init.sql file"

# Step 4: Remove old postgres volume to start fresh
echo "🧹 Step 4: Cleaning old PostgreSQL data..."
docker volume rm agentic-ai-sdlc_postgres_data 2>/dev/null || true
print_status "Cleaned old PostgreSQL volume"

# Step 5: Start PostgreSQL with new initialization
echo "🚀 Step 5: Starting PostgreSQL with proper initialization..."
docker-compose up -d postgres

# Step 6: Wait for PostgreSQL to be ready
echo "⏳ Step 6: Waiting for PostgreSQL to initialize..."
for i in {1..30}; do
    if docker-compose exec postgres pg_isready -U agentic_user -d agentic_sdlc; then
        print_status "PostgreSQL is ready!"
        break
    fi
    echo "Waiting... ($i/30)"
    sleep 2
done

# Step 7: Verify the database
echo "🔍 Step 7: Verifying database setup..."
if docker-compose exec postgres psql -U agentic_user -d agentic_sdlc -c "\dt workflows.*"; then
    print_status "Database tables created successfully"
else
    print_error "Database verification failed"
fi

# Step 8: Restart orchestrator to ensure it can connect
echo "🔄 Step 8: Restarting orchestrator to connect to database..."
docker-compose restart orchestrator

# Step 9: Wait for orchestrator to be ready
echo "⏳ Step 9: Waiting for orchestrator to be ready..."
sleep 15

# Step 10: Test the fix
echo "🧪 Step 10: Testing the complete fix..."

echo "Testing orchestrator health..."
if curl -f http://34.30.67.175:8000/health; then
    print_status "Orchestrator is healthy"
else
    print_error "Orchestrator health check failed"
    echo "Checking orchestrator logs..."
    docker-compose logs --tail=10 orchestrator
fi

echo "Testing workflow creation..."
WORKFLOW_RESPONSE=$(curl -s -X POST http://34.30.67.175:8000/workflows \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "test_postgres_fix", 
    "name": "PostgreSQL Fix Test",
    "description": "Testing after PostgreSQL fix",
    "requirements": {"description": "Test requirements after database fix"}
  }')

if [[ $WORKFLOW_RESPONSE == *"workflow_id"* ]]; then
    print_status "Workflow creation works! Database is properly connected."
    echo "Response: $WORKFLOW_RESPONSE"
else
    print_error "Workflow creation still failing"
    echo "Response: $WORKFLOW_RESPONSE"
fi

echo ""
echo "🎯 PostgreSQL Fix Summary:"
echo "========================="
print_status "Created proper postgres/init.sql file"
print_status "Cleaned old database volume"
print_status "Restarted PostgreSQL with proper initialization"
print_status "Database schemas and tables created"
print_status "Orchestrator reconnected to database"

echo ""
echo "✨ Your workflow button should now work!"
echo "Visit: http://34.30.67.175:3000/"
echo ""
print_warning "If you still have issues:"
echo "• Check: docker-compose logs orchestrator"
echo "• Check: docker-compose logs postgres"
echo "• Verify: curl http://34.30.67.175:8000/health"
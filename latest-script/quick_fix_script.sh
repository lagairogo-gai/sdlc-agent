#!/bin/bash

# Quick Fix Script for Workflow Issue at http://34.30.67.175:3000/
# This script addresses the frontend-backend connectivity problem

echo "🔧 Quick Fix for Agentic AI SDLC Workflow Issue"
echo "=============================================="

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

# Step 1: Check current system status
echo "🔍 Step 1: Checking current system status..."
docker-compose ps

# Step 2: Check orchestrator health
echo -e "\n🏥 Step 2: Testing orchestrator connectivity..."
if curl -f http://34.30.67.175:8000/health; then
    print_status "Orchestrator is responding"
else
    print_error "Orchestrator is not responding"
    echo "Checking orchestrator logs..."
    docker-compose logs --tail=20 orchestrator
fi

# Step 3: Update frontend configuration for your IP
echo -e "\n🔧 Step 3: Fixing frontend API configuration..."

# Create updated nginx.conf with your IP
cat > frontend/nginx.conf << 'EOF'
server {
    listen 3000;
    server_name localhost;

    # Frontend static files
    location / {
        root /usr/share/nginx/html;
        index index.html index.htm;
        try_files $uri $uri/ /index.html;
    }

    # API proxy to orchestrator
    location /api/ {
        proxy_pass http://34.30.67.175:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    # Direct API access (for workflow endpoints)
    location /workflows {
        proxy_pass http://34.30.67.175:8000/workflows;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }

    location /health {
        proxy_pass http://34.30.67.175:8000/health;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket endpoint
    location /ws {
        proxy_pass http://34.30.67.175:8000/ws;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # CORS headers
    add_header Access-Control-Allow-Origin *;
    add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS";
    add_header Access-Control-Allow-Headers "DNT,User-Agent,X-Requested-With,If-Modified-Since,Cache-Control,Content-Type,Range,Authorization";
}
EOF

print_status "Updated nginx configuration with correct API endpoint"

# Step 4: Update docker-compose.yml frontend section
echo -e "\n🔧 Step 4: Updating Docker Compose configuration..."

# Backup current docker-compose.yml
cp docker-compose.yml docker-compose.yml.backup

# Update the frontend service environment in docker-compose.yml
sed -i 's|REACT_APP_API_URL=.*|REACT_APP_API_URL=http://34.30.67.175:8000|g' docker-compose.yml
sed -i 's|REACT_APP_WS_URL=.*|REACT_APP_WS_URL=ws://34.30.67.175:8000/ws|g' docker-compose.yml

print_status "Updated Docker Compose with correct environment variables"

# Step 5: Rebuild and restart frontend
echo -e "\n🔨 Step 5: Rebuilding frontend container..."
docker-compose build frontend
print_status "Frontend container rebuilt"

echo -e "\n🚀 Step 6: Restarting frontend service..."
docker-compose up -d frontend
print_status "Frontend service restarted"

# Step 7: Wait for services to be ready
echo -e "\n⏳ Step 7: Waiting for services to be ready..."
sleep 15

# Step 8: Test the fix
echo -e "\n🧪 Step 8: Testing the fix..."

echo "Testing frontend accessibility..."
if curl -f http://34.30.67.175:3000/; then
    print_status "Frontend is accessible"
else
    print_error "Frontend is not accessible"
fi

echo "Testing API connectivity from frontend..."
if docker-compose exec frontend wget -qO- http://34.30.67.175:8000/health; then
    print_status "API connectivity from frontend works"
else
    print_error "API connectivity from frontend failed"
fi

# Step 9: Test workflow creation via API
echo -e "\n🔬 Step 9: Testing workflow creation directly..."
WORKFLOW_RESPONSE=$(curl -s -X POST http://34.30.67.175:8000/workflows \
  -H "Content-Type: application/json" \
  -d '{
    "project_id": "test_project_fix", 
    "name": "Test Fix Project",
    "description": "Testing the workflow fix",
    "requirements": {"description": "Test requirements for fix"}
  }')

if [[ $WORKFLOW_RESPONSE == *"workflow_id"* ]]; then
    print_status "Workflow creation API works correctly"
    echo "Response: $WORKFLOW_RESPONSE"
else
    print_error "Workflow creation API failed"
    echo "Response: $WORKFLOW_RESPONSE"
fi

# Step 10: Final instructions
echo -e "\n🎯 Step 10: Final Steps"
echo "=============================================="
print_status "Fix completed! Please try the following:"
echo ""
echo "1. 🌐 Visit: http://34.30.67.175:3000/"
echo "2. 📝 Fill in the project form:"
echo "   - Project Name: Test Project"
echo "   - Description: Testing workflow"
echo "   - Requirements: User authentication, API endpoints"
echo "3. 🚀 Click 'Start SDLC Workflow'"
echo "4. 👀 You should see the workflow appear and progress indicators"
echo ""
print_warning "If it still doesn't work:"
echo "• Check browser console (F12) for JavaScript errors"
echo "• Run: docker-compose logs frontend"
echo "• Run: docker-compose logs orchestrator"
echo ""
echo "🆘 Additional debugging commands:"
echo "• make health          - Check system health"
echo "• make logs-frontend   - View frontend logs"
echo "• make logs-orchestrator - View orchestrator logs"
echo "• make debug          - Run connectivity diagnostics"
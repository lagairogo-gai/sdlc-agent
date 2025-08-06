#!/bin/bash
set -e

echo "🚀 AI Monitoring System v5 - FIXED Enhanced Deployment"
echo "======================================================"

# Detect Docker Compose command
if command -v docker &> /dev/null && docker compose version &> /dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
elif command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
else
    echo "❌ Docker Compose not found"
    exit 1
fi

echo "✅ Using: $DOCKER_COMPOSE"

# Check environment
if [ ! -f .env ]; then
    echo "⚠️  Creating .env file from template..."
    cat > .env << EOF
DATADOG_API_KEY=demo_key
DATADOG_APP_KEY=demo_key  
PAGERDUTY_API_KEY=demo_key
SERVICENOW_INSTANCE=demo
SERVICENOW_USERNAME=demo
SERVICENOW_PASSWORD=demo
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=demo@company.com
SMTP_PASSWORD=demo
EOF
    echo "Created .env with demo values - please edit if needed!"
fi

# Clean slate - remove old containers and volumes if they exist
echo "🧹 Cleaning up existing deployment..."
$DOCKER_COMPOSE down -v --remove-orphans 2>/dev/null || true

# Remove any orphaned containers
docker container prune -f 2>/dev/null || true

# Build frontend first if it exists
if [ -d "frontend" ] && [ ! -d "frontend/build" ]; then
    echo "🎨 Building frontend..."
    if [ -f "scripts/build-frontend.sh" ]; then
        chmod +x scripts/build-frontend.sh
        ./scripts/build-frontend.sh
    else
        echo "⚠️  No frontend build script found, frontend may not work properly"
    fi
fi

# Build with enhancements
echo "🏗️  Building FIXED enhanced system (this may take a few minutes)..."
$DOCKER_COMPOSE build --no-cache

echo "🚀 Starting FIXED enhanced services..."
$DOCKER_COMPOSE up -d

# Enhanced health check with better timing
echo "⏳ Waiting for FIXED enhanced services to be ready..."
sleep 45

echo "🔍 Running FIXED enhanced health checks..."

# Check services with better error handling
for i in {1..30}; do
    echo "Health check attempt $i/30..."
    
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ FIXED Enhanced AI Monitoring System is ready!"
        break
    fi
    
    if [ $i -eq 30 ]; then
        echo "❌ System failed to start after 30 attempts"
        echo "📋 Showing last 50 lines of logs:"
        $DOCKER_COMPOSE logs --tail=50 ai-monitoring
        echo ""
        echo "🐛 Debug information:"
        echo "Container status:"
        $DOCKER_COMPOSE ps
        echo ""
        echo "Port binding:"
        docker port $(docker ps --format "table {{.Names}}" | grep ai-monitoring | head -1) 2>/dev/null || echo "No port info available"
        exit 1
    fi
    
    echo "Waiting for system startup... ($i/30)"
    sleep 5
done

# Test enhanced features with better error handling
echo "🧪 Testing FIXED enhanced features..."
sleep 10

# Test the API endpoints with timeout and better error handling
echo "Testing FIXED enhanced API endpoints..."

test_endpoint() {
    local name="$1"
    local url="$2"
    echo -n "  Testing $name... "
    
    if timeout 10 curl -f "$url" > /dev/null 2>&1; then
        echo "✅ OK"
        return 0
    else
        echo "❌ FAILED"
        return 1
    fi
}

test_endpoint "Dashboard Stats" "http://localhost:8000/api/dashboard/stats"
test_endpoint "Enhanced Agents" "http://localhost:8000/api/agents"
test_endpoint "Incidents API" "http://localhost:8000/api/incidents"
test_endpoint "Health Check" "http://localhost:8000/health"

echo ""
echo "🎉 FIXED ENHANCED DEPLOYMENT SUCCESSFUL!"
echo "========================================"
echo ""
echo "🆕 FIXED FEATURES AVAILABLE:"
echo "  🔄 Real-time incident workflow execution"
echo "  📊 Live agent progress tracking with progress bars"
echo "  📝 FIXED: Detailed console logs for each agent"
echo "  🔗 WebSocket real-time updates"
echo "  📱 FIXED: Interactive agent dashboard with working click-to-view logs"
echo "  📈 Comprehensive incident history and analytics"
echo "  🧠 Model Context Protocol (MCP) integration"
echo "  🤝 Agent-to-Agent (A2A) communication"
echo ""
echo "📊 Access Points:"
echo "  🌐 FIXED Enhanced Dashboard:  http://localhost:8000"
echo "  💚 Health Check:             http://localhost:8000/health"
echo "  📊 Dashboard Stats:          http://localhost:8000/api/dashboard/stats"
echo "  🤖 Agent Details:            http://localhost:8000/api/agents"
echo "  📋 Incident History:         http://localhost:8000/api/incidents"
echo "  📚 API Documentation:        http://localhost:8000/api/docs"
echo ""
echo "🧪 Try the FIXED Enhanced Features:"
echo "  1. Click 'Check For Incident' to see agents work in real-time"
echo "  2. Watch live progress bars as each agent executes"
echo "  3. FIXED: Click on agent tiles to view detailed execution logs"
echo "  4. FIXED: View detailed console logs for each agent"
echo "  5. FIXED: Click 'View Logs' buttons in incident details"
echo "  6. See complete incident workflow from start to resolution"
echo "  7. Check MCP Context Protocol status"
echo "  8. Monitor Agent-to-Agent communications"
echo ""
echo "🔧 Management Commands:"
echo "  View logs:    $DOCKER_COMPOSE logs -f ai-monitoring"
echo "  Stop system:  $DOCKER_COMPOSE down"
echo "  Restart:      $DOCKER_COMPOSE restart"
echo "  Clean reset:  $DOCKER_COMPOSE down -v && $DOCKER_COMPOSE up -d"
echo ""
echo "🌟 Your FIXED AI Monitoring System now has WORKING DETAILED CONSOLE LOGS!"
echo "🎯 Click any agent in the dashboard to view their detailed execution logs!"
echo ""
echo "🆘 If you encounter issues:"
echo "  1. Check logs: $DOCKER_COMPOSE logs ai-monitoring"
echo "  2. Restart: $DOCKER_COMPOSE restart"
echo "  3. Full reset: $DOCKER_COMPOSE down -v && $DOCKER_COMPOSE up -d"
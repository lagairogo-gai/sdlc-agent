#!/bin/bash

echo "🔧 AI Monitoring System - Frontend Rebuild & Redeploy"
echo "====================================================="

# Check if Docker Compose is available
if command -v docker compose &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
elif command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
else
    echo "❌ Docker Compose not found"
    exit 1
fi

echo "✅ Using: $DOCKER_COMPOSE"

# Step 1: Check if frontend source exists
echo ""
echo "🔍 Step 1: Checking frontend source..."
if [ ! -f "frontend/src/App.js" ]; then
    echo "❌ frontend/src/App.js not found!"
    echo "Please make sure you've replaced frontend/src/App.js with the fixed version"
    exit 1
fi

if [ ! -f "frontend/package.json" ]; then
    echo "❌ frontend/package.json not found!"
    exit 1
fi

echo "✅ Frontend source files found"

# Step 2: Stop existing containers
echo ""
echo "🛑 Step 2: Stopping existing containers..."
$DOCKER_COMPOSE down

# Step 3: Clean old builds
echo ""
echo "🧹 Step 3: Cleaning old builds..."
rm -rf frontend/build/
echo "✅ Removed old frontend build"

# Step 4: Build frontend
echo ""
echo "🎨 Step 4: Building frontend with new changes..."
cd frontend

# Check if node_modules exists, if not install
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
    if [ $? -ne 0 ]; then
        echo "❌ npm install failed"
        cd ..
        exit 1
    fi
fi

# Build frontend
echo "🏗️  Building React frontend..."
npm run build
if [ $? -ne 0 ]; then
    echo "❌ Frontend build failed"
    cd ..
    exit 1
fi

cd ..
echo "✅ Frontend built successfully"

# Step 5: Verify build exists
if [ ! -d "frontend/build" ]; then
    echo "❌ Frontend build directory not found after build"
    exit 1
fi

echo "✅ Build directory confirmed: frontend/build/"

# Step 6: Rebuild Docker containers with new frontend
echo ""
echo "🐳 Step 6: Rebuilding Docker containers..."
$DOCKER_COMPOSE build --no-cache ai-monitoring
if [ $? -ne 0 ]; then
    echo "❌ Docker build failed"
    exit 1
fi

echo "✅ Docker containers rebuilt with new frontend"

# Step 7: Start services
echo ""
echo "🚀 Step 7: Starting services..."
$DOCKER_COMPOSE up -d

# Step 8: Wait and health check
echo ""
echo "⏳ Step 8: Waiting for services to be ready..."
sleep 30

# Health check loop
echo "🔍 Performing health checks..."
for i in {1..20}; do
    echo "Health check attempt $i/20..."
    
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ System is healthy!"
        break
    fi
    
    if [ $i -eq 20 ]; then
        echo "❌ System failed health check after 20 attempts"
        echo "📋 Showing logs:"
        $DOCKER_COMPOSE logs --tail=30 ai-monitoring
        exit 1
    fi
    
    echo "Waiting... ($i/20)"
    sleep 5
done

# Step 9: Test frontend
echo ""
echo "🧪 Step 9: Testing frontend changes..."
sleep 5

# Test if we can access the main page
echo "Testing main page..."
if curl -s http://localhost:8000 | grep -q "OpsIntellect"; then
    echo "✅ Main page loads with new title"
else
    echo "⚠️  Main page might not have latest changes"
fi

# Test API endpoints
echo "Testing API endpoints..."
test_endpoint() {
    local name="$1"
    local url="$2"
    echo -n "  Testing $name... "
    
    if timeout 10 curl -f "$url" > /dev/null 2>&1; then
        echo "✅"
        return 0
    else
        echo "❌"
        return 1
    fi
}

test_endpoint "Dashboard Stats" "http://localhost:8000/api/dashboard/stats"
test_endpoint "Agents API" "http://localhost:8000/api/agents"
test_endpoint "Incidents API" "http://localhost:8000/api/incidents"

echo ""
echo "🎉 FRONTEND REBUILD COMPLETE!"
echo "============================="
echo ""
echo "🌐 Access your updated system:"
echo "  Frontend: http://localhost:8000"
echo "  Health:   http://localhost:8000/health"
echo "  API:      http://localhost:8000/api/docs"
echo ""
echo "🧪 Test the fixes:"
echo "  1. Open http://localhost:8000"
echo "  2. You should see: 'OpsIntellect - MCP + A2A + Detailed Logs AI System'"
echo "  3. Click 'Check For Incident'"
echo "  4. Click any agent tile → Should show detailed console logs"
echo "  5. Click any incident → Click 'View Logs' → Should work"
echo ""
echo "🔧 If issues persist:"
echo "  Check logs: $DOCKER_COMPOSE logs ai-monitoring"
echo "  Restart:    $DOCKER_COMPOSE restart"
echo ""
echo "Your frontend changes should now be visible! 🎯"
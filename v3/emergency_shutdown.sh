#!/bin/bash

echo "🚨 EMERGENCY SHUTDOWN - Stop Everything on Port 8000"
echo "===================================================="

echo ""
echo "⚠️  WARNING: This script will aggressively stop services that might"
echo "be serving content on port 8000. Only run if you're sure!"
echo ""
echo -n "Continue with emergency shutdown? (yes/no): "
read -r CONFIRM

if [[ "$CONFIRM" != "yes" ]]; then
    echo "❌ Aborted by user"
    exit 1
fi

echo ""
echo "🛑 STEP 1: Stop all web servers"
echo "==============================="

echo "Stopping nginx..."
sudo systemctl stop nginx 2>/dev/null && echo "✅ Nginx stopped" || echo "ℹ️ Nginx not running"

echo "Stopping apache2..."
sudo systemctl stop apache2 2>/dev/null && echo "✅ Apache2 stopped" || echo "ℹ️ Apache2 not running"

echo "Stopping httpd..."
sudo systemctl stop httpd 2>/dev/null && echo "✅ Httpd stopped" || echo "ℹ️ Httpd not running"

echo ""
echo "🛑 STEP 2: Kill processes aggressively"
echo "====================================="

echo "Killing all Python processes..."
sudo pkill -f python && echo "✅ Python processes killed" || echo "ℹ️ No Python processes to kill"

echo "Killing uvicorn specifically..."
sudo pkill -f uvicorn && echo "✅ Uvicorn killed" || echo "ℹ️ No uvicorn processes"

echo "Killing any processes with 'monitoring' in command..."
sudo pkill -f monitoring && echo "✅ Monitoring processes killed" || echo "ℹ️ No monitoring processes"

echo "Killing any processes with 'main.py'..."
sudo pkill -f main.py && echo "✅ main.py processes killed" || echo "ℹ️ No main.py processes"

echo ""
echo "🛑 STEP 3: Clear iptables rules"
echo "==============================="

echo "Flushing iptables INPUT chain..."
sudo iptables -F INPUT

echo "Flushing iptables FORWARD chain..."
sudo iptables -F FORWARD

echo "Flushing iptables OUTPUT chain..."
sudo iptables -F OUTPUT

echo "Flushing iptables NAT table..."
sudo iptables -t nat -F

echo "✅ All iptables rules cleared"

echo ""
echo "🛑 STEP 4: Stop Docker completely"
echo "================================"

echo "Stopping all Docker containers..."
docker stop $(docker ps -aq) 2>/dev/null && echo "✅ All containers stopped" || echo "ℹ️ No containers to stop"

echo "Removing all Docker containers..."
docker rm $(docker ps -aq) 2>/dev/null && echo "✅ All containers removed" || echo "ℹ️ No containers to remove"

echo "Pruning Docker system..."
docker system prune -af

echo ""
echo "🛑 STEP 5: Disable auto-restart services"
echo "======================================="

echo "Disabling any monitoring services..."
sudo systemctl disable ai-monitoring 2>/dev/null || echo "ℹ️ No ai-monitoring service"

echo "Stopping Docker daemon temporarily..."
sudo systemctl stop docker && echo "✅ Docker daemon stopped" || echo "⚠️ Could not stop Docker"

echo ""
echo "🧪 STEP 6: Test if port 8000 is finally free"
echo "==========================================="

sleep 5

echo "Testing localhost:8000..."
if curl -s http://localhost:8000 >/dev/null 2>&1; then
    echo "❌ localhost:8000 STILL responding"
else
    echo "✅ localhost:8000 not responding"
fi

echo "Testing 127.0.0.1:8000..."
if curl -s http://127.0.0.1:8000 >/dev/null 2>&1; then
    echo "❌ 127.0.0.1:8000 STILL responding"
else
    echo "✅ 127.0.0.1:8000 not responding"
fi

echo "Testing external 34.70.172.113:8000..."
if curl -s http://34.70.172.113:8000 >/dev/null 2>&1; then
    echo "❌ 34.70.172.113:8000 STILL responding (might be cloud load balancer)"
else
    echo "✅ 34.70.172.113:8000 not responding"
fi

echo ""
echo "🔍 STEP 7: Final process check"
echo "============================="

echo "Any remaining processes on port 8000:"
sudo lsof -i :8000 2>/dev/null || echo "✅ No processes on port 8000"

echo "Any remaining Python processes:"
ps aux | grep python | grep -v grep | head -5

echo ""
echo "🚀 STEP 8: Recovery instructions"
echo "==============================="

echo ""
echo "If 34.70.172.113:8000 is STILL responding after all this:"
echo ""
echo "1. 🌐 It's definitely a cloud load balancer or external proxy"
echo "2. 🔧 Check your cloud provider console (AWS, GCP, Azure, etc.)"
echo "3. 🎯 Look for:"
echo "   - Load Balancers pointing to your server"
echo "   - Auto Scaling Groups"
echo "   - Kubernetes clusters"
echo "   - Docker Swarm services"
echo "   - CI/CD pipelines auto-deploying"

echo ""
echo "If 34.70.172.113:8000 is NOW not responding:"
echo ""
echo "✅ Success! Port 8000 is finally free"
echo ""
echo "Now restart Docker and deploy your updated app:"
echo "  sudo systemctl start docker"
echo "  cd frontend && npm install && npm run build && cd .."
echo "  docker compose build --no-cache"
echo "  docker compose up -d"

echo ""
echo "🔄 To undo this emergency shutdown:"
echo "  sudo systemctl start nginx"
echo "  sudo systemctl start apache2"
echo "  sudo systemctl start docker"
echo "  # Restore iptables rules if you had custom ones"

echo ""
echo "Emergency shutdown complete! 🎯"
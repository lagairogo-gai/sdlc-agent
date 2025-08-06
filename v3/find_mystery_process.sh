#!/bin/bash

echo "🕵️ Mystery Process Detective - Finding What's Running on Port 8000"
echo "=================================================================="

echo ""
echo "🔍 STEP 1: COMPREHENSIVE PORT CHECK"
echo "===================================="

echo ""
echo "Checking ALL processes on port 8000..."
sudo lsof -i :8000 2>/dev/null || echo "lsof shows nothing on 8000"

echo ""
echo "Checking with ss command..."
sudo ss -tulpn | grep :8000 || echo "ss shows nothing on 8000"

echo ""
echo "Checking with fuser..."
sudo fuser 8000/tcp 2>/dev/null || echo "fuser shows nothing on 8000"

echo ""
echo "🔍 STEP 2: FINDING PYTHON/UVICORN PROCESSES"
echo "==========================================="

echo ""
echo "Looking for Python processes..."
ps aux | grep python | grep -v grep

echo ""
echo "Looking for uvicorn processes..."
ps aux | grep uvicorn | grep -v grep

echo ""
echo "Looking for any process with 'main.py' or 'monitoring'..."
ps aux | grep -E "(main\.py|monitoring)" | grep -v grep

echo ""
echo "🔍 STEP 3: CHECKING ALL PYTHON PROCESSES ON ANY PORT"
echo "===================================================="

echo ""
echo "All Python processes with network connections..."
sudo netstat -tulpn | grep python

echo ""
echo "🔍 STEP 4: DOCKER INVESTIGATION"
echo "==============================="

echo ""
echo "ALL Docker containers (including stopped)..."
docker ps -a

echo ""
echo "Docker networks..."
docker network ls

echo ""
echo "Any Docker containers that might be running on different networks..."
docker container ls --all --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"

echo ""
echo "🔍 STEP 5: SYSTEM INVESTIGATION"
echo "==============================="

echo ""
echo "Any service running on 8000 (systemd, etc.)..."
sudo systemctl --type=service --state=active | grep -i monitoring

echo ""
echo "Checking /etc/systemd/system/ for any monitoring services..."
ls -la /etc/systemd/system/ | grep -i monitoring

echo ""
echo "🔍 STEP 6: PROCESS TREE ANALYSIS"
echo "================================"

echo ""
echo "Full process tree to see what might be running..."
pstree -p | grep -A5 -B5 -E "(python|uvicorn|8000)"

echo ""
echo "🔍 STEP 7: BACKGROUND/NOHUP PROCESSES"
echo "====================================="

echo ""
echo "Checking for background processes..."
jobs -l

echo ""
echo "Checking for nohup processes..."
ps aux | grep nohup | grep -v grep

echo ""
echo "Checking for screen/tmux sessions..."
screen -ls 2>/dev/null || echo "No screen sessions"
tmux list-sessions 2>/dev/null || echo "No tmux sessions"

echo ""
echo "🔍 STEP 8: WEB SERVER CHECK"
echo "==========================="

echo ""
echo "Checking what's actually responding on 34.70.172.113:8000..."
echo "Response headers:"
curl -I http://34.70.172.113:8000/ 2>/dev/null || echo "Cannot connect"

echo ""
echo "Server identification:"
curl -s http://34.70.172.113:8000/health 2>/dev/null | head -5 || echo "No health endpoint"

echo ""
echo "🔍 STEP 9: ALTERNATIVE PORT BINDING"
echo "==================================="

echo ""
echo "Checking if something is bound to 0.0.0.0:8000 vs 127.0.0.1:8000..."
sudo netstat -tulpn | grep -E ":8000|:8080|:8888|:3000"

echo ""
echo "Checking all active network connections..."
sudo netstat -tulpn | grep LISTEN | sort

echo ""
echo "🔍 STEP 10: FIREWALL/PROXY CHECK"
echo "================================"

echo ""
echo "Checking if there's a proxy or firewall redirect..."
sudo iptables -L -n | grep 8000

echo ""
echo "Checking nginx/apache for any proxies..."
ps aux | grep -E "(nginx|apache)" | grep -v grep

echo ""
echo "📋 SUMMARY & RECOMMENDATIONS"
echo "============================"

echo ""
echo "Based on what we find above, here are the possibilities:"
echo ""
echo "1. 🐍 Python process running in background (most likely)"
echo "   - Look for uvicorn/python processes in the output above"
echo "   - Kill with: sudo kill -9 <PID>"
echo ""
echo "2. 🏃 Screen/tmux session with the app running"
echo "   - Check screen -ls and tmux list-sessions output"
echo "   - Attach and stop: screen -r <session> or tmux attach <session>"
echo ""
echo "3. 🔧 Systemd service"
echo "   - Look for monitoring services above"
echo "   - Stop with: sudo systemctl stop <service-name>"
echo ""
echo "4. 🔀 Port forwarding/proxy"
echo "   - Check iptables output above"
echo "   - Check for nginx/apache processes"
echo ""
echo "5. 🐳 Docker container on different network"
echo "   - Check all containers output above"
echo ""

echo ""
echo "🚀 NEXT STEPS:"
echo "1. Identify the mystery process from the output above"
echo "2. Stop it (kill PID, stop service, exit screen session, etc.)"
echo "3. Then proceed with your Docker rebuild"

echo ""
echo "If you find the process, run:"
echo "  sudo kill -9 <PID>  # Replace <PID> with the actual process ID"
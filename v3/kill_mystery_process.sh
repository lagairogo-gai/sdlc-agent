#!/bin/bash

echo "🔪 Kill Mystery AI Monitoring Process"
echo "====================================="

echo ""
echo "This script will find and kill any running AI monitoring processes"
echo "that might be serving the web app on port 8000."

echo ""
echo "🔍 Searching for processes..."

# Find Python processes that might be the monitoring system
echo ""
echo "Looking for Python/uvicorn processes..."
PYTHON_PROCESSES=$(ps aux | grep -E "(python.*main\.py|uvicorn|monitoring)" | grep -v grep)

if [ ! -z "$PYTHON_PROCESSES" ]; then
    echo "Found Python processes:"
    echo "$PYTHON_PROCESSES"
    echo ""
    
    # Extract PIDs
    PIDS=$(echo "$PYTHON_PROCESSES" | awk '{print $2}')
    
    for PID in $PIDS; do
        echo "Process PID $PID details:"
        ps -p $PID -o pid,ppid,cmd --no-headers 2>/dev/null
        
        echo -n "Kill this process? (y/n): "
        read -r RESPONSE
        
        if [[ "$RESPONSE" =~ ^[Yy]$ ]]; then
            echo "Killing process $PID..."
            sudo kill -9 $PID
            echo "✅ Process $PID killed"
        else
            echo "⏭️ Skipping process $PID"
        fi
        echo ""
    done
else
    echo "No Python processes found with main.py, uvicorn, or monitoring"
fi

# Check for processes on port 8000
echo ""
echo "🔍 Looking for ANY processes on port 8000..."
LSOF_OUTPUT=$(sudo lsof -i :8000 2>/dev/null)

if [ ! -z "$LSOF_OUTPUT" ]; then
    echo "Found processes on port 8000:"
    echo "$LSOF_OUTPUT"
    echo ""
    
    # Extract PIDs from lsof
    PIDS=$(echo "$LSOF_OUTPUT" | grep -v COMMAND | awk '{print $2}' | sort -u)
    
    for PID in $PIDS; do
        if [[ "$PID" =~ ^[0-9]+$ ]]; then
            echo "Process on port 8000 - PID $PID:"
            ps -p $PID -o pid,ppid,cmd --no-headers 2>/dev/null
            
            echo -n "Kill this process? (y/n): "
            read -r RESPONSE
            
            if [[ "$RESPONSE" =~ ^[Yy]$ ]]; then
                echo "Killing process $PID..."
                sudo kill -9 $PID
                echo "✅ Process $PID killed"
            else
                echo "⏭️ Skipping process $PID"
            fi
            echo ""
        fi
    done
else
    echo "No processes found specifically on port 8000"
fi

# Check for screen sessions
echo ""
echo "🔍 Checking for screen sessions..."
SCREEN_SESSIONS=$(screen -ls 2>/dev/null | grep -i monitoring || screen -ls 2>/dev/null | grep Detached)

if [ ! -z "$SCREEN_SESSIONS" ]; then
    echo "Found screen sessions:"
    echo "$SCREEN_SESSIONS"
    echo ""
    echo "You may need to manually check these sessions:"
    echo "  screen -ls"
    echo "  screen -r <session_name>"
    echo "  # Then exit the session"
else
    echo "No screen sessions found"
fi

# Check for tmux sessions
echo ""
echo "🔍 Checking for tmux sessions..."
TMUX_SESSIONS=$(tmux list-sessions 2>/dev/null | grep -i monitoring || tmux list-sessions 2>/dev/null)

if [ ! -z "$TMUX_SESSIONS" ]; then
    echo "Found tmux sessions:"
    echo "$TMUX_SESSIONS"
    echo ""
    echo "You may need to manually check these sessions:"
    echo "  tmux list-sessions"
    echo "  tmux attach-session -t <session_name>"
    echo "  # Then exit the session"
else
    echo "No tmux sessions found"
fi

# Kill common process patterns
echo ""
echo "🔪 Nuclear option - killing common patterns..."
echo -n "Kill ALL Python processes containing 'main.py' or 'monitoring'? (y/n): "
read -r RESPONSE

if [[ "$RESPONSE" =~ ^[Yy]$ ]]; then
    echo "Killing all matching processes..."
    
    # Kill processes by pattern
    sudo pkill -f "python.*main.py"
    sudo pkill -f "uvicorn"
    sudo pkill -f "monitoring"
    
    echo "✅ Killed processes by pattern"
else
    echo "⏭️ Skipped nuclear option"
fi

echo ""
echo "🧪 Testing if port 8000 is now free..."
sleep 2

if curl -s http://localhost:8000 >/dev/null 2>&1; then
    echo "❌ Something is STILL running on port 8000"
    echo ""
    echo "Manual investigation needed:"
    echo "1. Run: sudo lsof -i :8000"
    echo "2. Run: sudo netstat -tulpn | grep :8000"
    echo "3. Check: ps aux | grep python"
elif curl -s http://34.70.172.113:8000 >/dev/null 2>&1; then
    echo "❌ Something is STILL accessible on 34.70.172.113:8000"
    echo "This might be a proxy, load balancer, or external service"
else
    echo "✅ Port 8000 appears to be free now!"
    echo ""
    echo "🚀 You can now proceed with your Docker rebuild:"
    echo "   cd frontend && npm install && npm run build && cd .."
    echo "   docker compose build --no-cache"
    echo "   docker compose up -d"
fi

echo ""
echo "🔍 Current process check:"
echo "Processes still containing 'python' or 'monitoring':"
ps aux | grep -E "(python|monitoring)" | grep -v grep | head -5
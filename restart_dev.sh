#!/bin/bash

echo "🔄 Restarting RAG Web App..."

# Kill any existing processes
echo "Stopping existing processes..."
pkill -f "python.*app.py" || true
pkill -f "npm.*start" || true

# Wait a moment
sleep 2

# Start backend
echo "🚀 Starting FastAPI backend..."
cd /Users/navinhemani/Desktop/rag_web_react
python app.py &
BACKEND_PID=$!

# Wait for backend to start
echo "⏳ Waiting for backend..."
sleep 8

# Check if backend is running
if curl -s http://localhost:8000/api/health > /dev/null; then
    echo "✅ Backend is running"
else
    echo "⏳ Still starting... checking again..."
    sleep 3
    if curl -s http://localhost:8000/api/health > /dev/null; then
        echo "✅ Backend is running"
    else
        echo "❌ Backend failed to start"
        exit 1
    fi
fi

echo "🎉 Application restarted!"
echo "📱 Visit: http://localhost:8000/react"
echo "📊 API Health: http://localhost:8000/api/health"
echo ""
echo "To stop: pkill -f 'python.*app.py'"

#!/bin/bash

echo "🔥 REBUILDING WITH REAL-TIME FORMATTING & COLLAPSIBLE SIDEBAR"
echo "==========================================================="

cd /Users/navinhemani/Desktop/rag_web_react

# Kill servers
echo "💀 Stopping old servers..."
pkill -f "python.*app.py" 2>/dev/null || true

# Build with fixes
echo "🔨 Building with streaming fixes..."
cd frontend
npm run build

echo "✅ Build complete with optimizations:"
echo "   ⚡ Real-time markdown formatting during streaming"
echo "   📱 Collapsible sidebar with chevron buttons"
echo "   🔗 Live citations appearing during streaming"
echo "   🚀 Optimized performance"

# Start server
echo "🚀 Starting optimized server..."
cd ../
python app.py &
SERVER_PID=$!

echo ""
echo "🎯 READY! Open: http://localhost:8000/react"
echo "📌 NEW FEATURES:"
echo "   • Click ← → buttons to collapse/expand sidebar"
echo "   • Streaming now shows formatting in real-time"
echo "   • Citations appear as text streams"
echo ""
echo "Press Ctrl+C to stop"

wait $SERVER_PID
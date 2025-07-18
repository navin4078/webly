#!/bin/bash

echo "🔧 Building React frontend for deployment..."

# Check if Node.js is available
if command -v node &> /dev/null; then
    echo "✅ Node.js found: $(node --version)"
    
    # Install frontend dependencies and build
    cd frontend
    
    echo "📦 Installing frontend dependencies..."
    npm install
    
    echo "🏗️  Building React app..."
    npm run build
    
    echo "✅ React build completed!"
    cd ..
else
    echo "⚠️  Node.js not found - skipping React build"
    echo "💡 React build will be skipped in deployment"
fi

echo "🚀 Build script completed!"
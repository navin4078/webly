# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a RAG (Retrieval-Augmented Generation) web application with a **React-only frontend** and **FastAPI backend**. The application scrapes websites, stores content in FAISS vector store, and provides AI-powered chat using Google Gemini 2.5 Flash with real-time updates via Socket.IO.

## Architecture

### Backend (`app.py`)
- **FastAPI** server with Socket.IO integration for real-time communication
- **WebScraperRAGAgentWithMemory** (`rag_agent.py`) - Advanced RAG implementation with LangGraph conversation memory
- **FAISS vector store** with SentenceTransformers embeddings for semantic search
- **Google Gemini 2.5 Flash** for AI responses with streaming support
- **Real-time scraping** with progress updates via WebSocket

### Frontend (`frontend/`)
- **React 18** application with modern hooks and concurrent features
- **Tailwind CSS** + **Framer Motion** for styling and animations
- **Socket.IO client** for real-time communication
- **React Markdown** with syntax highlighting for chat messages
- **Progressive Web App** capabilities

### Key Components
- `ScrapingPanel.js` - Website scraping interface with real-time progress
- `ChatInterface.js` - AI chat with streaming responses and message history
- `useSocket.js` - Custom hook managing Socket.IO connection and events

## Development Commands

### Backend Setup & Running
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Start FastAPI server (serves both API and React build)
python app.py
```

### Frontend Development
```bash
cd frontend

# Install dependencies
npm install

# Development server (hot reloading)
npm start  # Runs on http://localhost:3000

# Production build
npm run build

# Run tests
npm test
```

### Quick Development Workflow
1. **Backend**: `python app.py` (serves on http://localhost:8000)
2. **Frontend Dev**: `cd frontend && npm start` (development server on http://localhost:3000)
3. **Production Build**: `cd frontend && npm run build` then restart backend

## Environment Configuration

### Required Environment Variables
```bash
# .env file in root directory
GOOGLE_API_KEY=your_google_api_key_here  # Required for Gemini AI

# Optional
ENVIRONMENT=development
LOG_LEVEL=INFO
```

### Frontend Environment Variables
```bash
# frontend/.env
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000
REACT_APP_DEBUG_MODE=true
```

## Key Technical Details

### RAG Implementation
- Uses **LangGraph** for conversation memory and state management
- **SentenceTransformers** for embeddings (all-MiniLM-L6-v2)
- **FAISS** vector store with cosine similarity search
- **Async web scraping** with configurable depth and concurrency
- **BeautifulSoup4** for HTML parsing with content cleaning

### Real-time Features
- **Socket.IO** events: scraping progress, typing indicators, connection status
- **Streaming responses** for AI chat with Server-Sent Events
- **WebSocket fallback** for enhanced connection reliability

### API Endpoints
```
POST /api/scrape          # Start website scraping
GET  /api/status/{id}     # Get scraping status
POST /api/chat            # Send chat message (with streaming)
GET  /api/history/{id}    # Get conversation history
POST /api/reset/{id}      # Reset scraping session
GET  /api/health          # Health check
```

## Development Notes

### Frontend Package.json Scripts
- `npm start` - Development server with source maps disabled
- `npm run build` - Production build with optimizations
- `npm test` - Run test suite
- `npm run eject` - Eject from Create React App (not recommended)

### Backend Entry Points
- `http://localhost:8000` - React frontend (main interface)
- `http://localhost:8000/react` - React frontend (alias)
- `http://localhost:8000/api/docs` - FastAPI documentation

### Testing & Debugging
- Frontend: Set `REACT_APP_DEBUG_MODE=true` for detailed logging
- Backend: Set `LOG_LEVEL=DEBUG` for verbose logging
- No specific test commands found - use `npm test` for frontend React tests

### Common Development Tasks
- **Add new React component**: Create in `frontend/src/components/`, follow existing patterns with Tailwind CSS
- **Modify scraping logic**: Edit `rag_agent.py`, specifically `WebScraperRAGAgentWithMemory` class
- **Update API endpoints**: Add to `app.py`, follow FastAPI patterns with Pydantic models
- **Style changes**: Use Tailwind classes, modify `frontend/tailwind.config.js` for theme customization
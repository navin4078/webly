import os
import logging
import threading
import uuid
import time
import json
import asyncio
import re
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

# FastAPI imports
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Socket.IO for enhanced WebSocket support
import socketio

# RAG Agent
from rag_agent import WebScraperRAGAgentWithMemory, ScrapingConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Pydantic models
class ScrapeRequest(BaseModel):
    url: str
    crawl_depth: int = 2
    max_pages: int = 10
    max_concurrent: int = 5

class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default_conversation"
    stream: bool = True

class ScrapingStatus(BaseModel):
    status: str
    progress: float
    message: str
    total_pages: int = 0
    scraped_pages: int = 0
    start_time: float
    url: str
    stats: Optional[Dict] = None

# FastAPI app initialization
app = FastAPI(
    title="RAG Chat - React + FastAPI", 
    description="AI-Powered Website Intelligence with React Frontend",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Socket.IO server
sio = socketio.AsyncServer(async_mode='asgi', cors_allowed_origins="*")
socket_app = socketio.ASGIApp(sio, app)

# Templates
templates = Jinja2Templates(directory="templates")

# Create directories
for directory in ["templates", "uploads", "vector_store", "crawl_results", "junk"]:
    Path(directory).mkdir(exist_ok=True)

# Mount React build
react_build_path = Path("frontend/build")
if react_build_path.exists() and react_build_path.is_dir():
    logger.info("✅ React build found - mounting static files")
    
    # Mount static files correctly
    static_path = react_build_path / "static"
    if static_path.exists():
        app.mount("/static", StaticFiles(directory=str(static_path)), name="react-static")
    
    REACT_AVAILABLE = True
    logger.info(f"📁 React static files mounted from: {static_path}")
else:
    logger.warning("⚠️ React build not found")
    logger.info("💡 Run: cd frontend && npm run build")
    REACT_AVAILABLE = False

# Global storage
scraping_status: Dict[str, ScrapingStatus] = {}
scraping_status_lock = threading.Lock()

class RAGManager:
    def __init__(self):
        self.agents: Dict[str, WebScraperRAGAgentWithMemory] = {}
    
    def get_or_create_agent(self, session_id: str) -> WebScraperRAGAgentWithMemory:
        if session_id not in self.agents:
            logger.info(f"Creating RAG agent for session: {session_id}")
            self.agents[session_id] = WebScraperRAGAgentWithMemory()
        return self.agents[session_id]
    
    def remove_agent(self, session_id: str):
        if session_id in self.agents:
            logger.info(f"Removing RAG agent for session: {session_id}")
            del self.agents[session_id]
    
    async def broadcast_to_session(self, session_id: str, event: str, data: Dict):
        try:
            await sio.emit(event, data, room=session_id)
            logger.info(f"Broadcasted {event} to session {session_id}")
        except Exception as e:
            logger.error(f"Failed to broadcast to session {session_id}: {e}")

rag_manager = RAGManager()

# Socket.IO events
@sio.event
async def connect(sid, environ):
    logger.info(f"Socket.IO client connected: {sid}")
    await sio.emit('connected', {'message': 'Connected successfully', 'sid': sid}, room=sid)

@sio.event
async def disconnect(sid):
    logger.info(f"Socket.IO client disconnected: {sid}")

@sio.event
async def join_room(sid, data):
    session_id = data.get('session_id')
    if session_id:
        await sio.enter_room(sid, session_id)
        logger.info(f"Client {sid} joined room {session_id}")
        await sio.emit('room_joined', {'session_id': session_id}, room=sid)

@sio.event
async def ping(sid):
    await sio.emit('pong', {'message': 'Server is alive'}, room=sid)

# MAIN ROUTES
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Redirect to React app"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/react", response_class=HTMLResponse)
async def react_app(request: Request):
    """Serve React application"""
    if not REACT_AVAILABLE:
        return HTMLResponse(
            content="""
            <html>
                <head><title>React App Not Available</title></head>
                <body style="font-family: Arial, sans-serif; text-align: center; padding: 2rem;">
                    <h1>⚠️ React Frontend Not Available</h1>
                    <p>Please build the React app first:</p>
                    <pre style="background: #333; color: white; padding: 1rem; border-radius: 8px;">
cd frontend
npm run build</pre>
                </body>
            </html>
            """,
            status_code=503
        )
    
    react_index = react_build_path / "index.html"
    if react_index.exists():
        return FileResponse(react_index)
    else:
        raise HTTPException(status_code=404, detail="React app not found")

# Serve React assets
@app.get("/manifest.json")
async def get_manifest():
    manifest_path = react_build_path / "manifest.json"
    if manifest_path.exists():
        return FileResponse(manifest_path)
    raise HTTPException(status_code=404, detail="Manifest not found")

@app.get("/favicon.ico")
async def get_favicon():
    favicon_path = react_build_path / "favicon.ico"
    if favicon_path.exists():
        return FileResponse(favicon_path)
    raise HTTPException(status_code=404, detail="Favicon not found")

# API ROUTES
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "time": datetime.now().isoformat(),
        "react_available": REACT_AVAILABLE,
        "agents_count": len(rag_manager.agents)
    }

@app.get("/api/debug/{session_id}")
async def debug_session(session_id: str):
    """Debug endpoint to check session state"""
    try:
        rag_agent = rag_manager.get_or_create_agent(session_id)
        return {
            "session_id": session_id,
            "has_graph": rag_agent.graph is not None,
            "has_vector_store": rag_agent.vector_store is not None,
            "has_embeddings": rag_agent.embeddings is not None,
            "scraped_urls_count": len(rag_agent.scraped_urls) if hasattr(rag_agent, 'scraped_urls') else 0,
            "ready_for_chat": rag_agent.graph is not None and rag_agent.vector_store is not None
        }
    except Exception as e:
        return {
            "session_id": session_id,
            "error": str(e),
            "ready_for_chat": False
        }

@app.post("/api/scrape")
async def start_scraping(scrape_data: ScrapeRequest, request: Request):
    session_id = request.headers.get('X-Session-ID', str(uuid.uuid4()))
    logger.info(f"Scrape request from session: {session_id}")
    
    try:
        url = scrape_data.url.strip()
        if not url:
            raise HTTPException(status_code=400, detail="URL is required")
        
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Validate URL
        from urllib.parse import urlparse
        try:
            result = urlparse(url)
            if not all([result.scheme, result.netloc]):
                raise ValueError("Invalid URL")
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid URL format")
        
        logger.info(f"Starting scrape: {url}")
        
        config = ScrapingConfig(
            max_depth=scrape_data.crawl_depth,
            max_pages=scrape_data.max_pages,
            max_concurrent=scrape_data.max_concurrent,
            delay_range=(0.5, 2.0),
            respect_robots=False,
            use_sitemap=True
        )
        
        with scraping_status_lock:
            scraping_status[session_id] = ScrapingStatus(
                status='starting',
                progress=0,
                message='Initializing...',
                total_pages=0,
                scraped_pages=0,
                start_time=time.time(),
                url=url
            )
        
        await rag_manager.broadcast_to_session(session_id, 'scraping_update', scraping_status[session_id].dict())
        asyncio.create_task(scrape_background(session_id, url, config))
        
        return {"success": True, "session_id": session_id, "message": "Scraping started"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Scraping error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start scraping: {str(e)}")

async def scrape_background(session_id: str, url: str, config: ScrapingConfig):
    logger.info(f"Background scraping started: {session_id}")
    
    try:
        rag_agent = rag_manager.get_or_create_agent(session_id)
        
        with scraping_status_lock:
            scraping_status[session_id].status = 'scraping'
            scraping_status[session_id].progress = 10
            scraping_status[session_id].message = f'Scraping {url}...'
        
        await rag_manager.broadcast_to_session(session_id, 'scraping_update', scraping_status[session_id].dict())
        
        # Run scraping
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            documents = await asyncio.get_event_loop().run_in_executor(
                executor, 
                rag_agent.scrape_website, 
                url, 
                config.max_depth, 
                config.max_pages, 
                config.respect_robots
            )
        
        logger.info(f"Scraped {len(documents) if documents else 0} documents")
        
        if not documents:
            with scraping_status_lock:
                scraping_status[session_id].status = 'error'
                scraping_status[session_id].progress = 0
                scraping_status[session_id].message = 'No content scraped'
            
            await rag_manager.broadcast_to_session(session_id, 'scraping_update', scraping_status[session_id].dict())
            return
        
        # Process documents
        with scraping_status_lock:
            scraping_status[session_id].status = 'processing'
            scraping_status[session_id].progress = 50
            scraping_status[session_id].message = f'Processing {len(documents)} pages...'
            scraping_status[session_id].scraped_pages = len(documents)
        
        await rag_manager.broadcast_to_session(session_id, 'scraping_update', scraping_status[session_id].dict())
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        logger.info(f"📝 Split {len(documents)} documents into {len(texts)} chunks for session {session_id}")
        
        with scraping_status_lock:
            scraping_status[session_id].status = 'creating_vectors'
            scraping_status[session_id].progress = 75
            scraping_status[session_id].message = f'Creating vectors from {len(texts)} chunks...'
        
        await rag_manager.broadcast_to_session(session_id, 'scraping_update', scraping_status[session_id].dict())
        
        logger.info(f"🌍 Creating vector store for session {session_id}...")
        rag_agent.vector_store = FAISS.from_documents(documents=texts, embedding=rag_agent.embeddings)
        logger.info(f"🔗 Creating RAG chain for session {session_id}...")
        rag_agent._create_conversational_rag_chain()
        logger.info(f"🤖 Creating LangGraph agent for session {session_id}...")
        rag_agent._create_langgraph_agent()
        logger.info(f"✅ RAG agent fully initialized for session {session_id}")
        
        total_time = time.time() - scraping_status[session_id].start_time
        
        with scraping_status_lock:
            scraping_status[session_id].status = 'completed'
            scraping_status[session_id].progress = 100
            scraping_status[session_id].message = f'Ready! Processed {len(documents)} pages in {total_time:.1f}s'
        
        await rag_manager.broadcast_to_session(session_id, 'scraping_update', scraping_status[session_id].dict())
        await rag_manager.broadcast_to_session(session_id, 'scraping_completed', {
            'session_id': session_id,
            'pages_scraped': len(documents),
            'total_time': total_time,
            'ready_for_chat': True
        })
        
        # Verify RAG agent is ready
        final_agent = rag_manager.get_or_create_agent(session_id)
        logger.info(f"🔍 Final RAG agent check for {session_id}: graph={final_agent.graph is not None}, vector_store={final_agent.vector_store is not None}")
        
        logger.info(f"✅ Scraping completed successfully: {session_id}")
        
    except Exception as e:
        logger.error(f"❌ Scraping error for {session_id}: {str(e)}")
        logger.error(f"❌ Error type: {type(e).__name__}")
        import traceback
        logger.error(f"❌ Full traceback: {traceback.format_exc()}")
        
        with scraping_status_lock:
            if session_id in scraping_status:
                scraping_status[session_id].status = 'error'
                scraping_status[session_id].progress = 0
                scraping_status[session_id].message = f'Error: {str(e)}'
        
        await rag_manager.broadcast_to_session(session_id, 'scraping_update', scraping_status[session_id].dict())

@app.post("/api/chat")
async def chat(chat_data: ChatRequest, request: Request):
    session_id = request.headers.get('X-Session-ID', 'default_session')
    
    # Enhanced debugging
    logger.info(f"🔍 Chat request debug - Session: {session_id}")
    logger.info(f"🔍 Chat request debug - Data: {chat_data}")
    logger.info(f"🔍 Chat request debug - Headers: {dict(request.headers)}")
    
    try:
        message = chat_data.message.strip()
        thread_id = chat_data.thread_id
        
        if not message:
            logger.error(f"❌ Empty message received for session {session_id}")
            raise HTTPException(status_code=400, detail="Message is required")
        
        logger.info(f"📝 Processing message: '{message[:50]}...' for session: {session_id}")
        
        rag_agent = rag_manager.get_or_create_agent(session_id)
        
        if rag_agent.graph is None:
            logger.error(f"❌ No graph found for session {session_id} - website not scraped")
            raise HTTPException(status_code=400, detail="Please scrape a website first")
        
        logger.info(f"✅ RAG agent ready for session {session_id}, streaming: {chat_data.stream}")
        
        if chat_data.stream:
            return StreamingResponse(
                stream_response(rag_agent, message, thread_id, session_id),
                media_type='text/plain',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                    'X-Accel-Buffering': 'no'
                }
            )
        else:
            result = rag_agent.query_with_memory(message, thread_id)
            return {
                'success': True,
                'answer': result['answer'],
                'thread_id': result['thread_id']
            }
        
    except HTTPException as he:
        logger.error(f"❌ HTTP Exception in chat: {he.detail} (status: {he.status_code})")
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected chat error for session {session_id}: {str(e)}")
        logger.error(f"❌ Error type: {type(e).__name__}")
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

async def stream_response(rag_agent, message: str, thread_id: str, session_id: str):
    try:
        logger.info(f"Starting streaming for session: {session_id}")
        
        await rag_manager.broadcast_to_session(session_id, 'typing_start', {'thread_id': thread_id})
        
        sources = []
        print(f"🚀 DEBUG - Starting streaming for message: {message[:50]}...")
        for chunk in rag_agent.query_with_memory_streaming(message, thread_id):
            if chunk:
                # Check if this chunk contains sources
                if '__SOURCES__:' in chunk:
                    try:
                        sources_json = chunk.split('__SOURCES__:')[1]
                        sources = json.loads(sources_json)
                        print(f"🚀 DEBUG - Extracted sources from stream: {sources}")
                        continue  # Don't send this chunk to frontend
                    except Exception as e:
                        print(f"❌ DEBUG - Failed to parse sources: {e}")
                        pass  # If parsing fails, treat as regular chunk
                
                # Strip ANY remaining citation formats as safety net
                clean_chunk = re.sub(r'\[Source:[^\]]*\]', '', chunk)
                clean_chunk = re.sub(r'\[\d+\]', '', clean_chunk)
                clean_chunk = re.sub(r'\(https?://[^\)]+\)', '', clean_chunk)
                yield f"data: {json.dumps({'chunk': clean_chunk, 'type': 'chunk'})}\n\n"
                await asyncio.sleep(0.005)  # Reduced sleep for faster streaming
        
        # Send sources separately at the end
        print(f"🚀 DEBUG - Final sources to send: {sources}")
        if sources:
            yield f"data: {json.dumps({'sources': sources, 'type': 'sources'})}\n\n"
            print(f"🚀 DEBUG - Sent sources to frontend: {len(sources)} sources")
        else:
            print(f"⚠️ DEBUG - No sources found to send!")
        
        yield f"data: {json.dumps({'type': 'complete'})}\n\n"
        await rag_manager.broadcast_to_session(session_id, 'typing_end', {'thread_id': thread_id})
        
    except Exception as e:
        logger.error(f"Streaming error: {e}")
        yield f"data: {json.dumps({'error': str(e), 'type': 'error'})}\n\n"
        await rag_manager.broadcast_to_session(session_id, 'typing_end', {'thread_id': thread_id})

@app.get("/api/status/{session_id}")
async def get_status(session_id: str):
    with scraping_status_lock:
        if session_id in scraping_status:
            status_data = scraping_status[session_id].dict()
            # Add agent state info
            try:
                rag_agent = rag_manager.get_or_create_agent(session_id)
                status_data["agent_ready"] = rag_agent.graph is not None and rag_agent.vector_store is not None
            except:
                status_data["agent_ready"] = False
            return status_data
    raise HTTPException(status_code=404, detail="No status found")

@app.get("/api/history/{session_id}")
async def get_history(session_id: str, thread_id: str = "default_conversation"):
    try:
        rag_agent = rag_manager.get_or_create_agent(session_id)
        
        if not hasattr(rag_agent, 'get_conversation_history'):
            return {"history": []}
        
        messages = rag_agent.get_conversation_history(thread_id)
        history = []
        for msg in messages:
            if hasattr(msg, 'content'):
                history.append({
                    'type': 'human' if 'Human' in str(type(msg)) else 'ai',
                    'content': msg.content,
                    'timestamp': datetime.now().isoformat()
                })
        
        return {"history": history}
        
    except Exception as e:
        logger.error(f"History error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clear_history/{session_id}")
async def clear_history(session_id: str, thread_id: str = "default_conversation"):
    try:
        rag_agent = rag_manager.get_or_create_agent(session_id)
        
        if hasattr(rag_agent, 'clear_conversation_history'):
            rag_agent.clear_conversation_history(thread_id)
        
        return {"success": True, "message": "History cleared"}
        
    except Exception as e:
        logger.error(f"Clear history error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reset_scraping/{session_id}")
async def reset_scraping(session_id: str):
    try:
        with scraping_status_lock:
            if session_id in scraping_status:
                del scraping_status[session_id]
        
        if session_id in rag_manager.agents:
            del rag_manager.agents[session_id]
        
        return {"success": True, "message": "Reset successful"}
        
    except Exception as e:
        logger.error(f"Reset error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket
@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    try:
        await websocket.send_text(json.dumps({
            "type": "connection",
            "message": "Connected",
            "session_id": session_id
        }))
        
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "ping":
                await websocket.send_text(json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                }))
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(status_code=404, content={"error": "Not found"})

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    logger.error(f"500 error: {str(exc)}")
    return JSONResponse(status_code=500, content={"error": "Internal server error"})

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable (Railway sets this)
    port = int(os.environ.get("PORT", 8000))
    
    logger.info("=" * 60)
    logger.info("🚀 Starting RAG Chat - React + FastAPI")
    logger.info(f"📱 Main Interface: http://0.0.0.0:{port}")
    logger.info(f"⚛️  React App: http://0.0.0.0:{port}/react")
    logger.info(f"🧪 API Docs: http://0.0.0.0:{port}/api/docs")
    logger.info(f"💊 Health: http://0.0.0.0:{port}/api/health")
    if not REACT_AVAILABLE:
        logger.info("💡 To enable React: cd frontend && npm run build")
    logger.info("=" * 60)
    
    try:
        uvicorn.run(socket_app, host="0.0.0.0", port=port, log_level="info")
    except Exception as e:
        logger.error(f"Failed to start server: {e}")

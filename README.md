# RAG Web React - AI-Powered Website Intelligence

A comprehensive web application that combines **React frontend** with **FastAPI backend** for intelligent website scraping and AI-powered conversations using Retrieval-Augmented Generation (RAG).

## 🌟 Features Overview

### 🎨 **Modern React Frontend**
- **⚛️ Sleek, responsive UI** - Built with React 18, Tailwind CSS, and Framer Motion
- **🌐 Real-time updates** - Live scraping progress and typing indicators via Socket.IO

### 🚀 **Core Capabilities**
- **Advanced Website Scraping** - Multi-depth crawling with configurable parameters
- **Real-time Communication** - Socket.IO integration for live updates and typing indicators
- **AI-Powered Chat** - Stream responses from Gemini 2.5 Flash with conversation memory
- **Vector Search** - FAISS-powered semantic search across scraped content
- **Citation Support** - Automatic source citations in AI responses
- **Progress Tracking** - Real-time scraping progress with detailed statistics

### 🛠️ **Technical Stack**
- **Frontend**: React 18, Tailwind CSS, Framer Motion, Socket.IO Client
- **Backend**: FastAPI, LangChain, LangGraph, Socket.IO
- **AI**: Google Gemini 2.5 Flash, SentenceTransformers embeddings
- **Storage**: FAISS vector store, in-memory conversation state

## 📁 Project Structure

```
rag_web_react/
├── 📱 frontend/                 # React application
│   ├── src/
│   │   ├── components/          # React components
│   │   ├── hooks/               # Custom hooks (Socket.IO, etc.)
│   │   ├── utils/               # Utilities and API client
│   │   └── App.js               # Main React app
│   ├── public/                  # Static assets
│   ├── package.json             # React dependencies
│   └── README.md                # Frontend documentation
├── 🐍 Backend Files
│   ├── app.py                   # Enhanced FastAPI server
│   ├── rag_agent.py             # Advanced RAG implementation
│   └── requirements.txt         # Python dependencies
├── 📁 Data Directories
│   ├── templates/               # HTML templates
│   ├── uploads/                 # File uploads
│   ├── vector_store/            # FAISS vector storage
│   └── crawl_results/           # Scraping results
├── 🔧 Setup & Config
│   ├── setup.sh                 # Linux/Mac setup script
│   ├── setup.bat                # Windows setup script
│   ├── .env                     # Environment variables
│   └── README.md                # This file
```

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+** (recommended: Python 3.10+)
- **Node.js 16+** (for React frontend)
- **Google API Key** (for Gemini AI) - [Get it here](https://makersuite.google.com/app/apikey)

### 🔧 Automated Setup

**Linux/macOS:**
```bash
chmod +x setup.sh
./setup.sh
```

**Windows:**
```batch
setup.bat
```

### 🔨 Manual Setup

1. **Clone and navigate:**
   ```bash
   cd rag_web_react
   ```

2. **Backend setup:**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # or
   venv\Scripts\activate     # Windows
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Environment configuration:**
   ```bash
   # Create .env file
   cp .env.example .env
   # Edit .env and add your GOOGLE_API_KEY
   ```

4. **Frontend setup:**
   ```bash
   cd frontend
   npm install
   npm run build
   cd ..
   ```

5. **Start the application:**
   ```bash
   python app.py
   ```

## 🌍 Access Points

Once running, access the application at:

- **🏠 React Frontend**: http://localhost:8000
- **⚛️ React App (alias)**: http://localhost:8000/react
- **📚 API Documentation**: http://localhost:8000/api/docs
- **💊 Health Check**: http://localhost:8000/api/health

## 🎯 How to Use

### 1. **Access the React Interface**
Visit http://localhost:8000 to access the modern React frontend.

### 2. **Configure Website Scraping**
- Enter a website URL
- Choose scraping mode (single page or full crawl)
- Set crawl depth and maximum pages
- Configure crawl speed

### 3. **Start Scraping**
- Click "Start Scraping"
- Watch real-time progress updates
- Wait for completion notification

### 4. **Chat with Content**
- Ask questions about the scraped content
- Enjoy streaming responses with citations
- Use conversation memory for context

## ⚛️ React Frontend Features

### 🎨 **Modern UI Components**
- **Responsive Design** - Mobile-first approach with Tailwind CSS
- **Smooth Animations** - Framer Motion powered transitions
- **Real-time Updates** - Socket.IO integration for live data
- **Progressive Web App** - Installable with offline capabilities

### 🔧 **Advanced Chat Interface**
- **Markdown Rendering** - Rich text formatting with ReactMarkdown
- **Code Highlighting** - Syntax highlighting for code blocks
- **Message Streaming** - Real-time response streaming
- **Typing Indicators** - Visual feedback during AI processing
- **Copy to Clipboard** - Easy message copying functionality

### 📱 **Responsive Features**
- **Mobile Optimized** - Touch-friendly interface
- **Adaptive Layout** - Automatic layout adjustments
- **Accessibility** - WCAG compliant design
- **Dark Mode Ready** - Prepared for theme switching

## 🔌 API Integration

### 🌐 **RESTful Endpoints**
```
POST /api/scrape          # Start website scraping
GET  /api/status/{id}     # Get scraping status  
POST /api/chat            # Send chat message
GET  /api/history/{id}    # Get conversation history
POST /api/reset/{id}      # Reset scraping session
```

### 📡 **Real-time Communication**
- **Socket.IO Events** - Live scraping updates and typing indicators
- **Server-Sent Events** - Streaming chat responses
- **WebSocket Fallback** - Enhanced connection reliability

## 🔧 Development

### 🐍 **Backend Development**
```bash
# Start backend only
source venv/bin/activate
python app.py
```

### ⚛️ **Frontend Development**
```bash
# Start React development server (separate terminal)
cd frontend
npm start
# Visit http://localhost:3000 for hot reloading
```

### 🔄 **Full Development Workflow**
1. **Backend**: `python app.py` (serves React build + API)
2. **Frontend**: `cd frontend && npm start` (development server)
3. **Build**: `cd frontend && npm run build` (update production build)

## 🎛️ Configuration

### 🔐 **Environment Variables**
```env
# Required
GOOGLE_API_KEY=your_google_api_key_here

# Optional
ENVIRONMENT=development
LOG_LEVEL=INFO
```

### ⚛️ **React Configuration**
```env
# frontend/.env
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000
REACT_APP_DEBUG_MODE=true
```

### 🐍 **Scraping Configuration**
```python
# Configurable in the UI or via ScrapingConfig
max_depth=2          # How deep to crawl
max_pages=10         # Maximum pages to scrape
max_concurrent=3     # Concurrent request limit
respect_robots=False # Bypass robots.txt (default)
use_sitemap=True     # Use sitemap for discovery
```

## 🎨 Customization

### 🎭 **Frontend Theming**
- **Tailwind CSS** - Utility-first styling
- **Custom Components** - Modular React components
- **Animation System** - Framer Motion animations
- **Icon Library** - Lucide React icons

### 🔧 **Backend Extension**
- **RAG Chain** - Customizable LangChain pipeline
- **Vector Store** - FAISS with SentenceTransformers
- **Memory System** - LangGraph conversation memory
- **Scraping Engine** - Advanced async crawling

## 🐛 Troubleshooting

### ❌ **Common Issues**

**React build not found:**
```bash
cd frontend
npm install
npm run build
```

**Backend connection failed:**
- Check if port 8000 is available
- Verify GOOGLE_API_KEY is set correctly
- Check firewall/antivirus settings

**Socket.IO connection issues:**
- Ensure WebSocket support in browser
- Check proxy/firewall settings
- Verify CORS configuration

**Scraping failures:**
- Try different websites
- Adjust crawl speed (lower concurrent requests)
- Check internet connectivity

### 🔍 **Debug Mode**
Enable debug logging:
```env
# Backend
LOG_LEVEL=DEBUG

# Frontend  
REACT_APP_DEBUG_MODE=true
```

## 🚀 Deployment

### 🐳 **Docker Deployment** (Coming Soon)
```dockerfile
# Dockerfile with React build included
FROM python:3.10
# ... (Docker configuration)
```

### ☁️ **Cloud Deployment**
- **Backend**: Deploy FastAPI to Heroku, Railway, or DigitalOcean
- **Frontend**: Deploy React build to Vercel, Netlify, or included in backend
- **Environment**: Set GOOGLE_API_KEY in cloud environment

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### 🎯 **Areas for Contribution**
- **UI/UX improvements** in React frontend
- **Additional AI models** integration
- **Enhanced scraping** capabilities
- **Performance optimizations**
- **Documentation** improvements
- **Testing** coverage
- **Accessibility** enhancements

## 📝 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

### 🛠️ **Core Technologies**
- **[React](https://reactjs.org/)** - Frontend framework
- **[FastAPI](https://fastapi.tiangolo.com/)** - Backend framework
- **[LangChain](https://langchain.com/)** - AI/LLM framework
- **[Google Gemini](https://ai.google.dev/)** - AI model
- **[Socket.IO](https://socket.io/)** - Real-time communication

### 🎨 **UI/UX Libraries**
- **[Tailwind CSS](https://tailwindcss.com/)** - Styling framework
- **[Framer Motion](https://www.framer.com/motion/)** - Animation library
- **[Lucide React](https://lucide.dev/)** - Icon library
- **[React Hot Toast](https://react-hot-toast.com/)** - Notifications

### 🔧 **Development Tools**
- **[Vite](https://vitejs.dev/)** - Build tool
- **[ESLint](https://eslint.org/)** - Code linting
- **[Prettier](https://prettier.io/)** - Code formatting

## 📊 React Frontend Features

| Feature | Status | Description |
|---------|--------|-------------|
| **UI Design** | ✅ Modern, Animated | Built with React 18, Tailwind CSS, Framer Motion |
| **Real-time Updates** | ✅ Socket.IO | Live scraping progress and typing indicators |
| **Mobile Responsive** | ✅ Optimized | Touch-friendly interface with adaptive layout |
| **Chat Streaming** | ✅ Advanced | Real-time AI response streaming |
| **Markdown Support** | ✅ Full | Rich text formatting with ReactMarkdown |
| **Syntax Highlighting** | ✅ Advanced | Code blocks with syntax highlighting |
| **PWA Support** | ✅ Full | Installable with offline capabilities |
| **Accessibility** | ✅ Enhanced | WCAG compliant design |
| **Customization** | ✅ Extensive | Tailwind CSS theming system |

## 🎯 Roadmap

### 🔮 **Upcoming Features**
- [ ] **Multi-language Support** - Internationalization
- [ ] **User Authentication** - Login and session management  
- [ ] **File Upload** - PDF and document processing
- [ ] **Export Options** - Save conversations and data
- [ ] **Advanced Analytics** - Usage statistics and insights
- [ ] **Plugin System** - Extensible architecture
- [ ] **Collaborative Features** - Multi-user support
- [ ] **Voice Interface** - Speech-to-text and text-to-speech

### 🎨 **UI Enhancements**
- [ ] **Theme Switching** - Light/dark mode toggle
- [ ] **Custom Themes** - User-defined color schemes
- [ ] **Layout Options** - Different chat layouts
- [ ] **Accessibility Tools** - Enhanced screen reader support

### 🚀 **Performance Improvements**
- [ ] **Caching Strategy** - Intelligent content caching
- [ ] **Background Processing** - Queue-based scraping
- [ ] **CDN Integration** - Global content delivery
- [ ] **Database Backend** - Persistent storage options

---

**🚀 Built with ❤️ using React + FastAPI + AI**

*For detailed React frontend documentation, see [frontend/README.md](frontend/README.md)*

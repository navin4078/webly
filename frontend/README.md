# RAG Chat Frontend - React Application

A modern React frontend for the RAG (Retrieval-Augmented Generation) chat application, integrated with FastAPI backend for real-time website scraping and AI-powered conversations.

## 🚀 Features

- **Modern React UI** - Built with React 18, Framer Motion animations, and Tailwind CSS
- **Real-time Communication** - Socket.IO integration for live scraping updates and typing indicators
- **Responsive Design** - Mobile-first design that works on all devices
- **Advanced Chat Interface** - Markdown support, code syntax highlighting, and message streaming
- **Progressive Web App** - PWA capabilities with offline support and app-like experience
- **Website Scraping Panel** - Intuitive interface for configuring website scraping parameters
- **Error Handling** - Comprehensive error handling with user-friendly notifications
- **Performance Optimized** - Code splitting, lazy loading, and optimized bundle size

## 🛠️ Technology Stack

- **React 18** - Latest React with concurrent features
- **Tailwind CSS** - Utility-first CSS framework for styling
- **Framer Motion** - Animation library for smooth transitions
- **Socket.IO Client** - Real-time bidirectional communication
- **React Markdown** - Markdown rendering for chat messages
- **React Syntax Highlighter** - Code syntax highlighting
- **Axios** - HTTP client for API requests
- **React Hot Toast** - Beautiful toast notifications
- **Lucide React** - Modern icon library

## 📁 Project Structure

```
frontend/
├── public/
│   ├── index.html          # HTML template
│   ├── manifest.json       # PWA manifest
│   └── robots.txt          # SEO robots file
├── src/
│   ├── components/         # React components
│   │   ├── Header.js       # Application header
│   │   ├── ScrapingPanel.js # Website scraping interface
│   │   └── ChatInterface.js # Chat interface
│   ├── hooks/              # Custom React hooks
│   │   └── useSocket.js    # Socket.IO integration
│   ├── utils/              # Utility functions
│   │   ├── api.js          # API client configuration
│   │   └── helpers.js      # Helper functions
│   ├── App.js              # Main application component
│   ├── index.js            # React application entry point
│   └── index.css           # Global styles and Tailwind imports
├── package.json            # Dependencies and scripts
├── tailwind.config.js      # Tailwind CSS configuration
├── postcss.config.js       # PostCSS configuration
└── .env                    # Environment variables
```

## 🚀 Getting Started

### Prerequisites

- Node.js 16+ (recommended: Node.js 18+)
- npm or yarn package manager
- FastAPI backend running on `http://localhost:8000`

### Installation

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```

2. **Install dependencies:**
   ```bash
   npm install
   # or
   yarn install
   ```

3. **Configure environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env file with your configuration
   ```

4. **Start the development server:**
   ```bash
   npm start
   # or
   yarn start
   ```

5. **Open your browser:**
   Navigate to `http://localhost:3000`

### Environment Variables

Configure these variables in your `.env` file:

```env
REACT_APP_API_URL=http://localhost:8000    # FastAPI backend URL
REACT_APP_WS_URL=ws://localhost:8000       # WebSocket URL
REACT_APP_DEBUG_MODE=true                  # Enable debug logging
```

## 🏗️ Building for Production

1. **Build the application:**
   ```bash
   npm run build
   # or
   yarn build
   ```

2. **Serve the build locally (optional):**
   ```bash
   npx serve -s build
   ```

The build folder will contain the optimized production build ready for deployment.

## 🎨 Component Overview

### Header Component
- Displays application title and branding
- Shows real-time connection status with Socket.IO
- Animated background effects and responsive design

### ScrapingPanel Component
- Website URL input with validation
- Configurable scraping parameters (depth, pages, speed)
- Real-time scraping progress updates
- Status indicators and error handling

### ChatInterface Component
- Modern chat UI with message bubbles
- Markdown support for rich text formatting
- Code syntax highlighting for technical content
- Typing indicators and streaming message support
- Message actions (copy, clear conversation)

## 🔧 Available Scripts

```bash
npm start          # Start development server
npm run build      # Build for production
npm test           # Run tests
npm run eject      # Eject from Create React App (not recommended)
```

## 🎯 Key Features

### Real-time Updates
- Socket.IO integration for live scraping progress
- Typing indicators during AI responses
- Connection status monitoring
- Automatic reconnection handling

### Advanced Chat
- Streaming message responses
- Markdown rendering with custom components
- Code syntax highlighting for multiple languages
- Message history and conversation management
- Copy-to-clipboard functionality

### Responsive Design
- Mobile-first approach
- Adaptive layout for different screen sizes
- Touch-friendly interface
- Progressive Web App capabilities

### Error Handling
- Comprehensive error boundaries
- User-friendly error messages
- Network error handling
- Fallback UI states

## 🔌 Integration with FastAPI Backend

The frontend communicates with the FastAPI backend through:

1. **REST API** - For standard CRUD operations and scraping requests
2. **Socket.IO** - For real-time updates and live communication
3. **Server-Sent Events** - For streaming chat responses

### API Endpoints Used

```javascript
// Scraping
POST /scrape              # Start website scraping
GET /status/{session_id}  # Get scraping status
POST /reset_scraping/{session_id} # Reset scraping session

// Chat
POST /chat                # Send chat message (with streaming)
GET /history/{session_id} # Get conversation history
POST /clear_history/{session_id} # Clear conversation

// Health
GET /health              # Backend health check
GET /test                # Connection test
```

## 🎨 Customization

### Styling
- Modify `tailwind.config.js` for theme customization
- Update `src/index.css` for global styles
- Component-specific styles in individual component files

### Configuration
- Update `src/utils/api.js` for API configuration
- Modify `src/hooks/useSocket.js` for Socket.IO settings
- Environment variables in `.env` file

## 📱 Progressive Web App

The application includes PWA features:

- **App Manifest** - Makes the app installable
- **Service Worker** - Enables offline functionality (in production)
- **Responsive Design** - Optimized for mobile devices
- **App-like Experience** - Full-screen mode and splash screen

## 🐛 Troubleshooting

### Common Issues

1. **Backend Connection Failed**
   - Ensure FastAPI backend is running on the correct port
   - Check CORS configuration in backend
   - Verify environment variables

2. **Socket.IO Connection Issues**
   - Check WebSocket URL configuration
   - Verify firewall/proxy settings
   - Check browser console for errors

3. **Build Errors**
   - Clear node_modules and reinstall dependencies
   - Check for conflicting package versions
   - Update Node.js to latest stable version

### Debug Mode

Enable debug mode by setting `REACT_APP_DEBUG_MODE=true` in your `.env` file for detailed logging.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **React Team** - For the amazing React framework
- **Tailwind CSS** - For the utility-first CSS framework
- **Framer Motion** - For smooth animations
- **Socket.IO** - For real-time communication
- **Lucide** - For beautiful icons

---

**Built with ❤️ using React + FastAPI**

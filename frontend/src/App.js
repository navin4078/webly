import React, { useState, useRef, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import { Toaster } from 'react-hot-toast';
import Header from './components/Header';
import ScrapingPanel from './components/ScrapingPanel';
import ChatInterface from './components/ChatInterface';
import { useScrapingUpdates, useTypingIndicator } from './hooks/useSocket';
import { apiUtils } from './utils/api';
import { generateSessionId } from './utils/helpers';
import { toast } from 'react-hot-toast';

function App() {
  // Session management
  const [sessionId] = useState(() => {
    const stored = sessionStorage.getItem('sessionId');
    if (stored) {
      console.log('🔄 Using existing session ID:', stored);
      return stored;
    }
    
    const newId = generateSessionId();
    sessionStorage.setItem('sessionId', newId);
    console.log('🆔 Generated new session ID:', newId);
    return newId;
  });

  // Application state
  const [hasScrapedContent, setHasScrapedContent] = useState(false);
  const [backendHealth, setBackendHealth] = useState(null);
  const [showSidebar, setShowSidebar] = useState(true);
  
  // Socket connections and real-time updates
  const { 
    scrapingStatus, 
    isScrapingActive, 
    socket: scrapingSocket 
  } = useScrapingUpdates(sessionId);
  
  const { 
    isTyping, 
    socket: typingSocket 
  } = useTypingIndicator(sessionId);
  
  // Get connection status from one of the sockets (they should be the same)
  const connectionStatus = scrapingSocket?.connectionStatus || 'disconnected';
  const isConnected = scrapingSocket?.isConnected || false;

  // Check backend health on mount
  useEffect(() => {
    const checkBackend = async () => {
      try {
        console.log('🔍 Checking backend health with session:', sessionId);
        const health = await apiUtils.checkHealth();
        setBackendHealth(health);
        console.log('✅ Backend health check passed:', health);
      } catch (error) {
        console.error('❌ Backend health check failed:', error);
        setBackendHealth({ status: 'unhealthy', error: error.message });
        toast.error('Backend connection failed: ' + error.message);
      }
    };

    checkBackend();
  }, [sessionId]);

  // Update scraped content status based on scraping status
  useEffect(() => {
    if (scrapingStatus?.status === 'completed') {
      setHasScrapedContent(true);
    }
  }, [scrapingStatus]);

  // Handle scraping start
  const handleScrapingStart = useCallback(async (params) => {
    try {
      console.log('🚀 Starting scraping with params:', params);
      setHasScrapedContent(false);
      const result = await apiUtils.startScraping(params);
      console.log('✅ Scraping started successfully:', result);
      toast.success('Scraping started successfully!');
      return result;
    } catch (error) {
      console.error('❌ Failed to start scraping:', error);
      throw error;
    }
  }, []);

  // Handle scraping reset
  const handleScrapingReset = useCallback(async () => {
    try {
      await apiUtils.resetScrapingSession(sessionId);
      setHasScrapedContent(false);
      toast.success('Scraping session reset');
    } catch (error) {
      console.error('❌ Failed to reset scraping:', error);
      toast.error('Failed to reset scraping session');
    }
  }, [sessionId]);

  // Loading screen
  if (backendHealth === null) {
    return (
      <div className="h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="w-8 h-8 border-2 border-orange-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600">Connecting to backend...</p>
        </div>
      </div>
    );
  }

  // Error screen
  if (backendHealth?.status === 'unhealthy') {
    return (
      <div className="h-screen flex items-center justify-center bg-gray-50">
        <div className="text-center max-w-md">
          <div className="w-16 h-16 bg-red-500 rounded-full flex items-center justify-center mx-auto mb-4">
            <span className="text-white text-2xl">!</span>
          </div>
          <h2 className="text-xl font-semibold text-gray-900 mb-2">Connection Failed</h2>
          <p className="text-gray-600 mb-4">{backendHealth.error}</p>
          <button
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-orange-500 text-white rounded-lg hover:bg-orange-600"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="h-screen bg-white overflow-hidden">
      <Toaster 
        position="top-center"
        toastOptions={{
          duration: 3000,
          style: {
            background: '#fff',
            color: '#374151',
            border: '1px solid #d1d5db',
            borderRadius: '0.75rem',
            fontSize: '14px'
          }
        }}
      />
      
      {/* Header */}
      <Header 
        isConnected={isConnected}
        connectionStatus={connectionStatus}
        showSidebar={showSidebar}
        onToggleSidebar={() => setShowSidebar(!showSidebar)}
      />

      {/* Main Layout */}
      <div className="flex h-[calc(100vh-56px)]">
        {/* Sidebar - COLLAPSIBLE */}
        {showSidebar && (
          <div className="w-80 border-r border-gray-200 bg-gray-50 flex-shrink-0">
            <ScrapingPanel
              onScrapingStart={handleScrapingStart}
              scrapingStatus={scrapingStatus}
              isScrapingActive={isScrapingActive}
              onReset={handleScrapingReset}
            />
          </div>
        )}

        {/* Chat Area */}
        <div className="flex-1 flex flex-col overflow-hidden">
          <ChatInterface
            hasScrapedContent={hasScrapedContent}
            isTyping={isTyping}
            sessionId={sessionId}
            showSidebar={showSidebar}
            onToggleSidebar={() => setShowSidebar(!showSidebar)}
          />
        </div>
      </div>
    </div>
  );
}

export default App;